mod complex;

fn main() {

    let c = complex::Complex::new(0.3, 0.6);
    let n = complex::iterate(c);

    println!("c diverges: {}", true); // prints "c diverges: true"

}
