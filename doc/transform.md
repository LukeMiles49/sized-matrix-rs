Applying a transformation to the points of a square:

```rust
use sized_matrix::Matrix;

let points = [
	Matrix::vector([0.0, 0.0]),
	Matrix::vector([0.0, 1.0]),
	Matrix::vector([1.0, 1.0]),
	Matrix::vector([1.0, 0.0]),
];

let shear = Matrix::rows([
	[1.0, 0.3],
	[0.0, 1.0],
]);

let transformed = (shear * Matrix::from_vectors(points)).to_vectors();

assert_eq!(transformed, [
	Matrix::vector([0.0, 0.0]),
	Matrix::vector([0.3, 1.0]),
	Matrix::vector([1.3, 1.0]),
	Matrix::vector([1.0, 0.0]),
]);
```
