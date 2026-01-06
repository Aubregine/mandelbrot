use std::ops::{Add, Mul};

const MAX_ITER: usize = 256;

#[derive(Copy, Clone, Debug)]
pub struct Complex(f64, f64);

// We define addition for complex numbers
impl Add for Complex {
    type Output = Complex;
    fn add(self, other: Complex) -> Complex {
        Complex(self.0 + other.0, self.1 + other.1)
    }
}

// We define multiplication for complex numbers
impl Mul for Complex {
    type Output = Complex;
    fn mul(self, other: Complex) -> Complex {
        Complex(
            self.0 * other.0 - self.1 * other.1,
            self.0 * other.1 + self.1 * other.0
        )
    }
}

// We define the norm of a complex number
impl Complex {
    pub fn new(re: f64, im: f64) -> Complex {
        Complex(re, im)
    }

    fn norm(&self) -> f64 {
        (self.0 * self.0 + self.1 * self.1).sqrt()
    }
}

pub fn iterate(c: Complex) -> usize {
    let mut z = Complex(0.0, 0.0);
    for n in 0..MAX_ITER {
        z = z * z + c;
        if z.norm() > 2.0 {
            return n;
        }
    }
    MAX_ITER
}