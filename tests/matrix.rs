use sized_matrix::{Matrix, Transpose};
use higher_order_functions::{Init, Map, Zip, Section};

use core::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::panic::{catch_unwind, UnwindSafe};

use num_traits::{
	One, Zero,
	Inv, Pow,
	MulAdd, MulAddAssign,
};

fn assert_panics<R, F: FnOnce() -> R + UnwindSafe>(f: F) {
	assert!(catch_unwind(f).is_err());
}

#[test]
fn init_matrix() {
	let matrix = Matrix::<_, 3, 5>::init(|[row, col]| (row, col));
	
	for row in 0..3 {
		for col in 0..5 {
			assert_eq!(matrix[[row, col]], (row, col));
		}
	}
}

#[test]
fn matrix_rows() {
	let matrix = Matrix::rows([
		[(0, 0), (0, 1)],
		[(1, 0), (1, 1)],
		[(2, 0), (2, 1)],
	]);
	
	assert_eq!(matrix, Matrix::<_, 3, 2>::init(|[row, col]| (row, col)));
}

#[test]
fn matrix_cols() {
	let matrix = Matrix::cols([
		[(0, 0), (1, 0), (2, 0)],
		[(0, 1), (1, 1), (2, 1)],
	]);
	
	assert_eq!(matrix, Matrix::<_, 3, 2>::init(|[row, col]| (row, col)));
}

#[test]
fn matrix_from_vectors() {
	let matrix = Matrix::from_vectors([
		Matrix::vector([(0, 0), (1, 0), (2, 0)]),
		Matrix::vector([(0, 1), (1, 1), (2, 1)]),
	]);
	
	assert_eq!(matrix, Matrix::<_, 3, 2>::init(|[row, col]| (row, col)));
}

#[test]
fn matrix_map_double() {
	let matrix = Matrix::rows([[1, 4, 5], [2, 3, 6]]);
	
	assert_eq!(matrix.map(|x| x * 2), Matrix::rows([[2, 8, 10], [4, 6, 12]]));
}

#[test]
fn matrix_map_cast() {
	let matrix = Matrix::rows([[1, 4, 5], [2, 3, 6]]);
	
	assert_eq!(matrix.map(f64::from), Matrix::rows([[1.0, 4.0, 5.0], [2.0, 3.0, 6.0]]));
}

#[test]
fn empty_matrix_section() {
	let m: Matrix<(), 0, 0> = Matrix::rows([]);
	
	let mat: Matrix<_, 0, 0> = m.section([0, 0]);
	
	assert_eq!(mat, Matrix::rows([]));
}

#[test]
fn matrix_empty_section() {
	let m = Matrix::<_, 3, 5>::init(|[row, col]| (row, col));
	
	assert_eq!(m.section([1, 2]), Matrix::<_, 0, 0>::rows([]));
	assert_eq!(m.section([1, 2]), Matrix::<_, 2, 0>::cols([]));
	assert_eq!(m.section([1, 2]), Matrix::<_, 0, 2>::rows([]));
}

#[test]
fn matrix_empty_out_of_bounds() {
	let m = Matrix::<_, 3, 5>::init(|[row, col]| (row, col));
	
	assert_panics(|| m.section([4, 2]) as Matrix<_, 0, 0>);
	assert_panics(|| m.section([1, 6]) as Matrix<_, 0, 0>);
}

#[test]
fn matrix_middle_section() {
	let m = Matrix::<_, 3, 5>::init(|[row, col]| (row, col));
	
	assert_eq!(m.section([1, 1]), Matrix::rows([[(1, 1), (1, 2), (1, 3)]]));
}

#[test]
fn matrix_corner_sections() {
	let m = Matrix::<_, 3, 5>::init(|[row, col]| (row, col));
	
	assert_eq!(m.section([0, 0]), Matrix::rows([[(0, 0), (0, 1)], [(1, 0), (1, 1)]]));
	assert_eq!(m.section([0, 3]), Matrix::rows([[(0, 3), (0, 4)], [(1, 3), (1, 4)]]));
	assert_eq!(m.section([1, 3]), Matrix::rows([[(1, 3), (1, 4)], [(2, 3), (2, 4)]]));
	assert_eq!(m.section([1, 0]), Matrix::rows([[(1, 0), (1, 1)], [(2, 0), (2, 1)]]));
}

#[test]
fn matrix_side_sections() {
	let m = Matrix::<_, 3, 5>::init(|[row, col]| (row, col));
	
	assert_eq!(m.section([0, 0]), Matrix::rows([[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]]));
	assert_eq!(m.section([0, 4]), Matrix::rows([[(0, 4)], [(1, 4)], [(2, 4)]]));
	assert_eq!(m.section([2, 0]), Matrix::rows([[(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)]]));
	assert_eq!(m.section([0, 0]), Matrix::rows([[(0, 0)], [(1, 0)], [(2, 0)]]));
}

#[test]
fn matrix_full_section() {
	let m = Matrix::<_, 3, 5>::init(|[row, col]| (row, col));
	
	assert_eq!(m.section([0, 0]), Matrix::<_, 3, 5>::init(|[row, col]| (row, col)));
}

#[test]
fn empty_matrix_zip() {
	let a: Matrix<(), 0, 0> = Matrix::rows([]);
	let b: Matrix<(), 0, 0> = Matrix::rows([]);
	
	let mat = a.zip(b, |_: (), _: ()| panic!("Shouldn't call zip function"));
	
	assert_eq!(mat, Matrix::rows([]));
}

#[test]
fn singleton_matrix_zip() {
	let a = Matrix::rows([[123]]);
	let b = Matrix::rows([[456]]);
	let mut called = false;
	
	let arr = a.zip(b, |a, b| {
		assert_eq!(a, 123);
		assert_eq!(b, 456);
		if called { panic!("Should only call zip function once"); }
		else { called = true; }
		321
	});
	
	assert!(called);
	assert_eq!(arr, Matrix::rows([[321]]));
}

#[test]
fn matrix_zip() {
	let a = Matrix::<_, 3, 5>::init(|[row, col]| (row, col, "a"));
	let b = Matrix::<_, 3, 5>::init(|[row, col]| (row, col, "b"));
	
	let arr = a.zip(b, |ax, bx| (ax, bx));
	
	assert_eq!(arr, Matrix::<_, 3, 5>::init(|[row, col]| ((row, col, "a"), (row, col, "b"))));
}

#[test]
fn matrix_zero() {
	let matrix = Matrix::<usize, 3, 5>::zero();
	
	assert_eq!(matrix, Matrix::<_, 3, 5>::init(|_| 0));
}

#[test]
fn matrix_one() {
	let matrix = Matrix::<usize, 3, 3>::one();
	
	assert_eq!(matrix, Matrix::<_, 3, 3>::init(|[row, col]| if row == col { 1 } else { 0 }));
}

#[test]
fn matrix_eq() {
	assert_eq!(Matrix::rows([[1, 2], [3, 4]]), Matrix::rows([[1, 2], [3, 4]]));
	assert_ne!(Matrix::rows([[1, 2], [3, 4]]), Matrix::rows([[1, 2], [3, 0]]));
	assert_ne!(Matrix::rows([[1, 2], [3, 4]]), Matrix::rows([[1, 2], [0, 4]]));
	assert_ne!(Matrix::rows([[1, 2], [3, 4]]), Matrix::rows([[1, 0], [3, 4]]));
	assert_ne!(Matrix::rows([[1, 2], [3, 4]]), Matrix::rows([[0, 2], [3, 4]]));
	assert_eq!(Matrix::<(), 0, 0>::rows([]), Matrix::<(), 0, 0>::rows([]));
}

#[test]
fn matrix_hash() {
	fn hash<T: Hash>(t: T) -> u64 {
		let mut hasher = DefaultHasher::new();
		t.hash(&mut hasher);
		hasher.finish()
	}
	
	// This isn't technically guaranteed to succeed, but I can't think of a better way to test it
	assert_eq!(hash(Matrix::rows([[1, 2], [3, 4]])), hash(Matrix::rows([[1, 2], [3, 4]])));
	assert_ne!(hash(Matrix::rows([[1, 2], [3, 4]])), hash(Matrix::rows([[1, 2], [3, 0]])));
	assert_ne!(hash(Matrix::rows([[1, 2], [3, 4]])), hash(Matrix::rows([[1, 2], [0, 4]])));
	assert_ne!(hash(Matrix::rows([[1, 2], [3, 4]])), hash(Matrix::rows([[1, 0], [3, 4]])));
	assert_ne!(hash(Matrix::rows([[1, 2], [3, 4]])), hash(Matrix::rows([[0, 2], [3, 4]])));
	assert_eq!(hash(Matrix::<(), 0, 0>::rows([])), hash(Matrix::<(), 0, 0>::rows([])));
}

#[test]
fn matrix_debug() {
	assert_eq!(
		format!("{:?}", Matrix::<_, 3, 2>::init(|[row, col]| (row, col))),
		"Matrix([[(0, 0), (0, 1)], [(1, 0), (1, 1)], [(2, 0), (2, 1)]])",
	);
	
	assert_eq!(format!("{:?}", Matrix::<(), 0, 0>::rows([])), "Matrix([])");
}

#[test]
fn matrix_default() {
	let matrix = Matrix::<usize, 3, 5>::default();
	
	assert_eq!(matrix, Matrix::<_, 3, 5>::init(|_| 0));
}

#[test]
fn matrix_transpose() {
	let matrix = Matrix::<_, 3, 5>::init(|[row, col]| (row, col)).transpose();
	
	assert_eq!(matrix, Matrix::<_, 5, 3>::init(|[row, col]| (col, row)));
}

#[test]
fn matrix_mul_scalar() {
	let matrix = Matrix::rows([[1, 4, 5], [2, 3, 6]]);
	
	assert_eq!(matrix * 2, Matrix::rows([[2, 8, 10], [4, 6, 12]]));
}

#[test]
fn matrix_mul_assign_scalar() {
	let mut matrix = Matrix::rows([[1, 4, 5], [2, 3, 6]]);
	
	matrix *= 2;
	
	assert_eq!(matrix, Matrix::rows([[2, 8, 10], [4, 6, 12]]));
}

#[test]
fn matrix_div_scalar() {
	let matrix = Matrix::rows([[1, 4, 5], [2, 3, 6]]);
	
	assert_eq!(matrix / 2, Matrix::rows([[0, 2, 2], [1, 1, 3]]));
}

#[test]
fn matrix_div_assign_scalar() {
	let mut matrix = Matrix::rows([[1, 4, 5], [2, 3, 6]]);
	
	matrix /= 2;
	
	assert_eq!(matrix, Matrix::rows([[0, 2, 2], [1, 1, 3]]));
}

#[test]
fn matrix_add() {
	let a = Matrix::rows([[1, 4, 5], [2, 3, 6]]);
	
	let b = Matrix::rows([[0, 1, 1], [2, 3, 5]]);
	
	assert_eq!(a + b, Matrix::rows([[1, 5, 6], [4, 6, 11]]));
}

#[test]
fn matrix_add_assign() {
	let mut a = Matrix::rows([[1, 4, 5], [2, 3, 6]]);
	
	let b = Matrix::rows([[0, 1, 1], [2, 3, 5]]);
	
	a += b;
	
	assert_eq!(a, Matrix::rows([[1, 5, 6], [4, 6, 11]]));
}

#[test]
fn matrix_sub() {
	let a = Matrix::rows([[1, 4, 5], [2, 3, 6]]);
	
	let b = Matrix::rows([[0, 1, 1], [2, 3, 5]]);
	
	assert_eq!(a - b, Matrix::rows([[1, 3, 4], [0, 0, 1]]));
}

#[test]
fn matrix_sub_assign() {
	let mut a = Matrix::rows([[1, 4, 5], [2, 3, 6]]);
	
	let b = Matrix::rows([[0, 1, 1], [2, 3, 5]]);
	
	a -= b;
	
	assert_eq!(a, Matrix::rows([[1, 3, 4], [0, 0, 1]]));
}

#[test]
fn matrix_neg() {
	let matrix = Matrix::rows([[1, -4, 5], [-2, 3, -6]]);
	
	assert_eq!(-matrix, Matrix::rows([[-1, 4, -5], [2, -3, 6]]));
}

#[test]
fn matrix_mul() {
	let a = Matrix::rows([[1, 4, 5], [2, 3, 6]]);
	
	let b = Matrix::rows([[0, 1], [1, 2], [3, 5]]);
	
	assert_eq!(a * b, Matrix::rows([[19, 34], [21, 38]]));
}

#[test]
fn matrix_mul_assign() {
	let mut a = Matrix::rows([[0, 1], [1, 2], [3, 5]]);
	
	let b = Matrix::rows([[2, 3], [3, 5]]);
	
	a *= b;
	
	assert_eq!(a, Matrix::rows([[3, 5], [8, 13], [21, 34]]));
}

#[test]
fn matrix_mul_add() {
	let a = Matrix::rows([[1, 4, 5], [2, 3, 6]]);
	
	let b = Matrix::rows([[0, 1], [1, 2], [3, 5]]);
	
	let c = Matrix::rows([[2, 3], [5, 7]]);
	
	assert_eq!(a.mul_add(b, c), Matrix::rows([[21, 37], [26, 45]]));
}

#[test]
fn matrix_mul_add_assign() {
	let mut a = Matrix::rows([[0, 1], [1, 2], [3, 5]]);
	
	let b = Matrix::rows([[2, 3], [3, 5]]);
	
	let c = Matrix::rows([[2, 3], [5, 7], [11, 13]]);
	
	a.mul_add_assign(b, c);
	
	assert_eq!(a, Matrix::rows([[5, 8], [13, 20], [32, 47]]));
}

#[test]
fn matrix_div() {
	let a = Matrix::rows([[3., 5.], [8., 13.], [21., 34.]]);
	
	let b = Matrix::rows([[2., 3.], [3., 5.]]);
	
	assert_eq!(a / b, Matrix::rows([[0., 1.], [1., 2.], [3., 5.]]));
}

#[test]
fn matrix_div_assign() {
	let mut a = Matrix::rows([[3., 5.], [8., 13.], [21., 34.]]);
	
	let b = Matrix::rows([[2., 3.], [3., 5.]]);
	
	a /= b;
	
	assert_eq!(a, Matrix::rows([[0., 1.], [1., 2.], [3., 5.]]));
}

#[test]
fn matrix_inv() {
	let matrix = Matrix::rows([[2., 3.], [3., 5.]]);
	
	assert_eq!(matrix.inv(), Matrix::rows([[5., -3.], [-3., 2.]]));
}

#[test]
fn matrix_pow() {
	let matrix = Matrix::rows([[2., 3.], [3., 5.]]);
	
	assert_eq!(matrix.pow(0), Matrix::rows([[1., 0.], [0., 1.]]));
	assert_eq!(matrix.pow(1), Matrix::rows([[2., 3.], [3., 5.]]));
	assert_eq!(matrix.pow(3), Matrix::rows([[89., 144.], [144., 233.]]));
	assert_eq!(matrix.pow(-1), Matrix::rows([[5., -3.], [-3., 2.]]));
	assert_eq!(matrix.pow(-3), Matrix::rows([[233., -144.], [-144., 89.]]));
}
