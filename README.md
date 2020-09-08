# sized_matrix

[Crate](https://crates.io/crates/sized_matrix)

[Documentation](https://docs.rs/sized_matrix)

[Repository](https://github.com/LukeMiles49/sized-matrix-rs)

[Changelog](https://github.com/LukeMiles49/sized-matrix-rs/blob/master/CHANGELOG.md)

Sized matrices using const generics for better type checking and performance.

```rust
use sized_matrix::{Matrix, Vector};

let a: Matrix<i32, 3, 4> = Matrix::rows([
	[ 1,  2,  3,  4],
	[ 5,  6,  7,  8],
	[ 9, 10, 11, 12],
]);

let b: Matrix<i32, 4, 2> = Matrix::rows([
	[ 0,  1],
	[ 1,  2],
	[ 3,  5],
	[ 8, 13],
]);

let c: Matrix<i32, 3, 2> = a * b;

assert_eq!(c, Matrix::rows([
	[ 43,  72],
	[ 91, 156],
	[139, 240],
]));

let d: Vector<i32, 2> = Matrix::vector([-1, 1]);

let e: Vector<i32, 3> = c * d;

assert_eq!(e, Matrix::vector([
	 29,
	 65,
	101,
]));
```

To use this, add it as a dependency to your Cargo.toml:
```toml
[dependencies]
sized_matrix = "^0.2.2"
```
