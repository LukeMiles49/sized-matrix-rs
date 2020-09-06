use super::{Matrix, Dot};

use higher_order_functions::Init;

use core::ops::{
	Mul,
	Index, IndexMut,
};

use num_traits::{MulAdd, Zero};

/// A special case type alias for Matrices with a single column.
pub type Vector<T, const M: usize> = Matrix<T, M, 1>;

impl<T, const M: usize> Vector<T, M> {
	/// Construct a column vector from an array.
	///
	/// # Examples
	///
	/// Constructing a `Vector`:
	///
	/// ```rust
	/// use sized_matrix::{Vector, Matrix};
	///
	/// let vector: Vector<i32, 3> = Vector::vector([1, 2, 3]);
	///
	/// assert_eq!(vector, Matrix::cols([[1, 2, 3]]));
	/// ```
	pub fn vector(contents: [T; M]) -> Self {
		Self::cols([contents])
	}
}

impl<T, const M: usize> Init<T, usize> for Vector<T, M> {
	fn init_with<F: FnMut(usize) -> T>(_: (), mut elem: F) -> Self {
		Self::init(|[row, _]: [usize; 2]| elem(row))
	}
}

impl<T, const M: usize> Index<usize> for Vector<T, M> {
	type Output = T;
	
	fn index(&self, index: usize) -> &T {
		&self[[index, 0]]
	}
}

impl<T, const M: usize> IndexMut<usize> for Vector<T, M> {
	fn index_mut(&mut self, index: usize) -> &mut T {
		&mut self[[0, index]]
	}
}

impl<TLhs: Copy, TRhs: Copy, TOutput, const M: usize> Dot<Vector<TRhs, M>> for Vector<TLhs, M> where
	TLhs: Mul<TRhs, Output = TOutput> + MulAdd<TRhs, TOutput, Output = TOutput>,
	TOutput: Zero,
{
	type Output = TOutput;
	
	fn dot(self, rhs: Vector<TRhs, M>) -> Self::Output {
		let mut result = Self::Output::zero();
		for k in 0..M {
			result = self[k].mul_add(rhs[k], result);
		}
		result
	}
}
