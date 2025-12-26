use std::fmt::{Debug, Formatter};
use std::ops::{Add, Mul};
use std::sync::mpsc::Receiver;
use macroquad::color::hsl_to_rgb;
use macroquad::prelude::*;
use rayon::prelude::*;
use Divergence::*;

const MAX_ITER: usize = 256;
const ITER_MULTIPLIER: f32 = 1.2;
const MAX_RADIUS: f64 = 2.0;
const MOVE_AMOUNT: f64 = 0.1;

/// Mandelbrot Viewer
/// DONE: parallelize the computation of the divergence
/// DONE: save the result of the computation to an image
/// DONE: compute the color of the background based on the result of the iteration
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

    let mut rx = parallel_draw_image(
        &canva_x,
        &canva_y,
        image_x as usize,
        image_y as usize,
        max_iter,
    );

    loop {
        clear_background(WHITE);
        if let Ok(img) = rx.try_recv() {
            texture.update(&img);
        }

        if is_key_pressed(KeyCode::Escape) {
            break;
        }

        if let Some(receiver) = handle_key_pressed(&mut max_iter, &mut center, &mut zoom, image_x, image_y) {
            rx = receiver;
        }

        draw_texture(&texture, 0.0, 0.0, WHITE);
        draw_rectangle(0.0, 0.0, 72.0, 24.0, WHITE);
        draw_text(&max_iter.to_string(), 10.0, 22.0, 20.0, BLACK);
        draw_rectangle(0.0, 26.0, 72.0, 24.0, WHITE);
        draw_text(&format!("{:.2e}x", 2.0/zoom), 10.0, 46.0, 20.0, BLACK);
        next_frame().await;
    }
}

fn handle_key_pressed(max_iter: &mut usize, center: &mut Complex, zoom: &mut f64, image_x: u16, image_y: u16) -> Option<Receiver<Image>> {
    let mut changed = false;

    if is_key_pressed(KeyCode::KpAdd) {
        *zoom /= 2.0;
        *max_iter = (*max_iter as f32 * ITER_MULTIPLIER) as usize; // this is also an issue
        changed = true;
    }
    if is_key_pressed(KeyCode::KpSubtract) {
        *zoom *= 2.0;
        *max_iter = (*max_iter as f32 / ITER_MULTIPLIER) as usize;
        changed = true;
    }
    if is_key_pressed(KeyCode::Up) {
        center.im -= MOVE_AMOUNT * *zoom;
        changed = true;
    }
    if is_key_pressed(KeyCode::Down) {
        center.im += MOVE_AMOUNT * *zoom;
        changed = true;
    }
    if is_key_pressed(KeyCode::Left) {
        center.re -= MOVE_AMOUNT * *zoom;
        changed = true;
    }
    if is_key_pressed(KeyCode::Right) {
        center.re += MOVE_AMOUNT * *zoom;
        changed = true;
    }

    if changed {
        let canva_x = Range { start: center.re - *zoom, end: center.re + *zoom };
        let canva_y = Range { start: center.im - *zoom, end: center.im + *zoom };
        Some(parallel_draw_image(
            &canva_x,
            &canva_y,
            image_x as usize,
            image_y as usize,
            *max_iter,
        ))
    } else { None }
}



fn parallel_draw_image(
    canva_x: &Range,
    canva_y: &Range,
    image_width: usize,
    image_height: usize,
    max_iter: usize,
) -> Receiver<Image> {
    let (tx, rx) = std::sync::mpsc::channel();
    let canva_x = canva_x.clone();
    let canva_y = canva_y.clone();
    let scale_x = canva_x.size() / (image_width as f64 - 1.0);
    let scale_y = canva_y.size() / (image_height as f64 - 1.0);


    std::thread::spawn(move || {
        let mut image = Image::gen_image_color(image_width as u16, image_height as u16, WHITE);
        let pixels = image.get_image_data_mut();
        pixels
            .par_chunks_mut(image_width)
            .enumerate()
            .for_each(|(y, row)| {
                // precision bottleneck
                let im: f64 = canva_y.start + y as f64 * scale_y;
                for (x, pixel) in row.iter_mut().enumerate() {
                    let re: f64 = canva_x.start + x as f64 * scale_x;

                    let x_minus_025 = re - 0.25;
                    let y2 = im * im;
                    let q = x_minus_025 * x_minus_025 + y2;

                    // optimization I found on internet, I haven't checked the math
                    let div = if q * (q + x_minus_025) < 0.25 * y2 || (im + 1.0) * (im + 1.0) + y2 < 0.0625 {
                        Never
                    } else {
                        iterate(Complex { re, im }, max_iter, MAX_RADIUS)
                    };

                    // directly set the raw pixel data
                    *pixel = match div {
                        After(j) => {
                            let val = j as u8;
                            [val, val, val, 255] // gery
                        },
                        Never => [0, 0, 0, 255], // black
                    }
                }
            });

        let _ = tx.send(image);
    });
    rx
}

#[derive(Copy, Clone)]
struct Range {
    start: f64,
    end: f64,
}

impl Range {
    fn size(&self) -> f64 {
        self.end - self.start
    }
}

#[derive(Clone)]
enum Divergence {
    After(usize),
    Never,
}

fn iterate(c: Complex, max_iter: usize, max_radius: f64) -> Divergence
{
    let mut z = c;
    for i in 0..max_iter {
        z = z * z + c;
        if z.module_sqr() > max_radius * max_radius {
            // let delta = (0.5 * z.module_sqr().log2() / max_radius.log2()).log2() as f32;
            // let t = (i as f32 + 1.0 - delta) / max_iter as f32;
            // let shade = t.powf(0.6);
            // return After(shade.min(1.0));
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
        write!(f, "{:?}+{:?}i", self.re, self.im)
    }
}

impl Mul for Complex {
    type Output = Self;
    fn mul(self, other: Self) -> Self::Output {
        Complex {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

impl Add for Complex {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Complex {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}