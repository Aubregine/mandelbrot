use std::fmt::{Debug, Formatter};
use std::ops::{Add, Mul, RangeInclusive};
use std::sync::{Arc, Mutex};
use std::sync::mpsc::Receiver;
use macroquad::prelude::*;
use rayon::prelude::*;
use Divergence::*;

// const SUBDIVISIONS_X: usize = 128;
const MAX_ITER: usize = 20;
const MAX_RADIUS: f64 = 500.0;

/// Mandelbrot Viewer
/// DONE: parallelize the computation of the divergence
/// TODO: save the result of the computation to an image
/// TODO: compute the color of the background based on the result of the iteration
#[macroquad::main("Mandelbrot Viewer")]
async fn main() {
    // let's say canva goes from -2 to 1 and -i to i
    let mut canva_x = -2.0..=1.0;
    let mut canva_y = -1.0..=1.0;

    let image_x = screen_width() as u16;
    let image_y = screen_height() as u16;

    let image = Image::gen_image_color(image_x, image_y, WHITE);
    let texture = Texture2D::from_image(&image);

    let image = Arc::new(Mutex::new(image));
    let mut rx = parallel_draw_image(
        image.clone(),
        &canva_x,
        &canva_y,
        image_x as usize,
        image_y as usize,
    );

    loop {
        clear_background(WHITE);
        if let Ok(new_image) = rx.try_recv() {
            let img = new_image.lock().unwrap();
            texture.update(&img);
            (canva_x, canva_y) = zoom(canva_x, canva_y, 0.01, 0.0035, 1.01);
            rx = parallel_draw_image(
                image.clone(),
                &canva_x,
                &canva_y,
                image_x as usize,
                image_y as usize,
            );
        }

        draw_texture(&texture, 0.0, 0.0, WHITE);
        next_frame().await;
    }
}

fn zoom(
    canva_x: RangeInclusive<f64>,
    canva_y: RangeInclusive<f64>,
    offset_x: f64,
    offset_y: f64,
    zoom_factor: f64
) -> (RangeInclusive<f64>, RangeInclusive<f64>){
    let new_canva_x = RangeInclusive::new(
        (canva_x.start() - offset_x) / zoom_factor,
        (canva_x.end() - offset_x) / zoom_factor
    );
    let new_canva_y = RangeInclusive::new(
        (canva_y.start() - offset_y) / zoom_factor,
        (canva_y.end() - offset_y) / zoom_factor
    );
    (new_canva_x, new_canva_y)
}

fn parallel_draw_image(
    image: Arc<Mutex<Image>>,
    canva_x: &RangeInclusive<f64>,
    canva_y: &RangeInclusive<f64>,
    image_width: usize,
    image_height: usize,
) -> Receiver<Arc<Mutex<Image>>> {
    let (tx, rx) = std::sync::mpsc::channel();
    let canva_x = canva_x.clone();
    let canva_y = canva_y.clone();
    std::thread::spawn(move || {
        (0..image_width * image_height)
            .into_par_iter()
            // cut in chunks
            .by_uniform_blocks(image_width)
            .map(|i| {
                let x = i % image_width;
                let y = i / image_width;

                // precision bottleneck
                let re: f64 = canva_x.start() + (x as f64 * canva_x.size() / image_width as f64);
                let im: f64 = canva_y.start() + (y as f64 * canva_y.size() / image_height as f64);

                iterate(Complex { re, im }, MAX_ITER, MAX_RADIUS)
            })
            // reunite the chunks
            .fold(Vec::new, |mut acc, div| { acc.push(div); acc})
            .reduce(Vec::new, |mut acc, mut chunk| { acc.append(&mut chunk); acc})
            // set the image pixels
            .iter().enumerate()
            .for_each(|(i, d)| {
                image.lock().unwrap().set_pixel((i % image_width) as u32, (i / image_width) as u32, match *d {
                    After(_) => WHITE,
                    Never => BLACK,
                });
            });
        let _ = tx.send(image);
    });
    rx
}

trait RangeExt {
    fn size(&self) -> f64;
}

impl RangeExt for RangeInclusive<f64> {
    fn size(&self) -> f64 {
        self.end() - self.start()
    }
}

enum Divergence {
    After(usize),
    Never,
}

fn iterate(c: Complex, max_iter: usize, max_radius: f64) -> Divergence {
    let mut z = c;
    for i in 0..max_iter {
        z = z * z + c;
        if z.module_sqr() > max_radius * max_radius {
            return After(i);
        }
    }
    Never
}

#[derive(Copy, Clone)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn module_sqr(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

impl Debug for Complex {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}+{}i", self.re, self.im)
    }
}

impl Mul for Complex {
    type Output = Complex;
    fn mul(self, other: Complex) -> Complex {
        Complex {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

impl Add for Complex {
    type Output = Complex;
    fn add(self, other: Complex) -> Complex {
        Complex {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}