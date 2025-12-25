use std::fmt::{Debug, Formatter};
use std::ops::{Add, AddAssign, Mul, Sub};
use std::process::Output;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::Receiver;
use macroquad::color::hsl_to_rgb;
use macroquad::prelude::*;
use rayon::prelude::*;
use Divergence::*;

const MAX_ITER: usize = 200;
const MAX_RADIUS: f64 = 5.0;

/// Mandelbrot Viewer
/// DONE: parallelize the computation of the divergence
/// DONE: save the result of the computation to an image
/// DONE: compute the color of the background based on the result of the iteration
/// TODO: implement zoom with wheel and mouse position
/// TODO: make zoom framerate independent
#[macroquad::main("Mandelbrot Viewer")]
async fn main() {
    let mut max_iter = MAX_ITER;
    let mut center = Complex { re: 0.0, im: 0.0 };
    let mut zoom = 2.0;
    let canva_x = Range { start: center.re - zoom, end: center.re + zoom };
    let canva_y = Range { start: center.im - zoom, end: center.im + zoom };

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
        max_iter,
    );

    loop {
        clear_background(WHITE);
        if let Ok(new_image) = rx.try_recv() {
            let img = new_image.lock().unwrap();
            texture.update(&img);
        }

        if is_key_pressed(KeyCode::Escape) {
            break;
        }

        if let Some(receiver) = handle_key_pressed(&mut max_iter, &mut center, &mut zoom, image_x, image_y, &image) {
            rx = receiver;
        }

        draw_texture(&texture, 0.0, 0.0, WHITE);
        next_frame().await;
    }
}

fn handle_key_pressed(max_iter: &mut usize, center: &mut Complex<f64>, zoom: &mut f64, image_x: u16, image_y: u16, image: &Arc<Mutex<Image>>) -> Option<Receiver<Arc<Mutex<Image>>>> {
    let mut changed = false;

    if is_key_pressed(KeyCode::KpAdd) {
        *zoom /= 1.2;
        *max_iter = (*max_iter as f32 * 1.12) as usize;
        changed = true;
    }
    if is_key_pressed(KeyCode::KpSubtract) {
        *zoom *= 1.2;
        *max_iter = (*max_iter as f32 * 0.88) as usize;
        changed = true;
    }
    if is_key_pressed(KeyCode::Up) {
        center.im -= 0.01;
        changed = true;
    }
    if is_key_pressed(KeyCode::Down) {
        center.im += 0.01;
        changed = true;
    }
    if is_key_pressed(KeyCode::Left) {
        center.re -= 0.01;
        changed = true;
    }
    if is_key_pressed(KeyCode::Right) {
        center.re += 0.01;
        changed = true;
    }

    if changed {
        let canva_x = Range { start: center.re - *zoom, end: center.re + *zoom };
        let canva_y = Range { start: center.im - *zoom, end: center.im + *zoom };
        Some(parallel_draw_image(
            image.clone(),
            &canva_x,
            &canva_y,
            image_x as usize,
            image_y as usize,
            *max_iter,
        ))
    } else { None }
}

fn parallel_draw_image(
    image: Arc<Mutex<Image>>,
    canva_x: &Range<f64>,
    canva_y: &Range<f64>,
    image_width: usize,
    image_height: usize,
    max_iter: usize,
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
                let re: f64 = canva_x.start + (x as f64 * canva_x.size() / (image_width - 1) as f64);
                let im: f64 = canva_y.start + (y as f64 * canva_y.size() / (image_height - 1) as f64);

                let div = iterate(Complex { re, im }, max_iter, MAX_RADIUS);

                image.lock().unwrap().set_pixel((i % image_width) as u32, (i / image_width) as u32, match div {
                    // After(j) => hsl_to_rgb(j, 1.0, 0.5),
                    After(j) => Color::new(j, j, j, 1.0),
                    Never => BLACK,
                });
            })
            // reunite the chunks
            .fold(Vec::new, |mut acc, div| { acc.push(div); acc})
            .reduce(Vec::new, |mut acc, mut chunk| { acc.append(&mut chunk); acc});
        let _ = tx.send(image);
    });
    rx
}

#[derive(Copy, Clone)]
struct Range<T> {
    start: T,
    end: T,
}

impl<T> Add<T> for Range<T> where T: Add<Output = T> + Copy {
    type Output = Self;
    fn add(self, offset: T) -> Self {
        Range { start: self.start + offset, end: self.end + offset }
    }
}

impl<T> AddAssign<T> for Range<T> where T: Add + AddAssign + Copy {
    fn add_assign(&mut self, offset: T) {
        self.start += offset;
        self.end += offset;
    }
}

impl Range<f64> {
    fn size(&self) -> f64 {
        self.end - self.start
    }
}

enum Divergence {
    After(f32),
    Never,
}

fn iterate<T>(c: Complex<T>, max_iter: usize, max_radius: f64) -> Divergence
where T: Add<Output = T> + Mul<Output = T> + Sub<Output = T> + Copy + PartialOrd + Into<f64>
{
    let mut z = c;
    for i in 0..max_iter {
        z = z * z + c;
        if z.module_sqr() > max_radius * max_radius {
            let delta = (0.5 * z.module_sqr().log2() / max_radius.log2()).log2() as f32;
            let t = (i as f32 + 1.0 - delta) / max_iter as f32;
            let shade = t.powf(0.6);
            return After(shade.min(1.0));
        }
    }
    Never
}

#[derive(Copy, Clone)]
struct Complex<T> {
    re: T,
    im: T,
}

impl<T> Complex<T> where T: Mul<Output = T> + Add<Output = T> + Copy + Into<f64> {
    fn module_sqr(&self) -> f64 {
        f64::try_from(self.re * self.re + self.im * self.im).unwrap_or(f64::INFINITY)
    }
}

impl<T> Debug for Complex<T> where T: Debug {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}+{:?}i", self.re, self.im)
    }
}

impl<T> Mul for Complex<T> where T: Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Copy {
    type Output = Self;
    fn mul(self, other: Self) -> Self::Output {
        Complex {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

impl<T> Add for Complex<T> where T: Add<Output = T> + Copy {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Complex {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}