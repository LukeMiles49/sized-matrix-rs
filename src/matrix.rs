use super::{Scalar, Transpose, Vector};

use higher_order_functions::{Init, Map, Zip, Section};

use core::ops::{
	Add, AddAssign,
	Sub, SubAssign,
	Neg,
	Mul, MulAssign,
	Div, DivAssign,
	Index, IndexMut,
};

use core::hash::{
	Hash, Hasher,
};

use core::fmt::{
	self,
	Debug, Formatter,
};

use num_traits::{
	One, Zero,
	Inv, Pow,
	MulAdd, MulAddAssign,
	PrimInt,
};

/// An `M` by `N` matrix of `T`s.
///
/// # Examples
///
/// Multiplying two matrices of different sizes:
///
/// ```rust
/// use sized_matrix::Matrix;
///
/// let a: Matrix<i32, 3, 4> = Matrix::rows([
///     [ 1,  2,  3,  4],
///     [ 5,  6,  7,  8],
///     [ 9, 10, 11, 12],
/// ]);
///
/// let b: Matrix<i32, 4, 2> = Matrix::rows([
///     [ 0,  1],
///     [ 1,  2],
///     [ 3,  5],
///     [ 8, 13],
/// ]);
///
/// let c: Matrix<i32, 3, 2> = a * b;
///
/// assert_eq!(c, Matrix::rows([
///     [ 43,  72],
///     [ 91, 156],
///     [139, 240],
/// ]));
/// ```
#[derive(Copy, Clone)]
pub struct Matrix<T, const M: usize, const N: usize> {
	contents: [[T; M]; N],
}

impl<T, const M: usize, const N: usize> Matrix<T, M, N> {
	fn new(contents: [[T; M]; N]) -> Self {
		Self {
			contents,
		}
	}
	
	/// Construct a matrix from an array of columns of values.
	///
	/// # Examples
	///
	/// Constructing a matrix from columns:
	///
	/// ```rust
	/// use sized_matrix::Matrix;
	///
	/// let matrix = Matrix::cols([
	///     [1, 3],
	///     [2, 4],
	/// ]);
	///
	/// assert_eq!(matrix[[0, 0]], 1);
	/// assert_eq!(matrix[[0, 1]], 2);
	/// assert_eq!(matrix[[1, 0]], 3);
	/// assert_eq!(matrix[[1, 1]], 4);
	/// ```
	pub fn cols(cols: [[T; M]; N]) -> Self {
		Matrix::new(cols)
	}
	
	/// Construct a matrix from an array of rows of values.
	///
	/// # Examples
	///
	/// Constructing a matrix from rows:
	///
	/// ```rust
	/// use sized_matrix::Matrix;
	///
	/// let matrix = Matrix::rows([
	///     [1, 2],
	///     [3, 4],
	/// ]);
	///
	/// assert_eq!(matrix[[0, 0]], 1);
	/// assert_eq!(matrix[[0, 1]], 2);
	/// assert_eq!(matrix[[1, 0]], 3);
	/// assert_eq!(matrix[[1, 1]], 4);
	/// ```
	pub fn rows(rows: [[T; N]; M]) -> Self {
		Matrix::new(rows).transpose()
	}
	
	/// Construct a matrix from an array of column vectors.
	///
	/// # Examples
	///
	#[doc(include = "../doc/transform.md")]
	pub fn from_vectors(vectors: [Vector<T, M>; N]) -> Self {
		Self {
			contents: vectors.map(|v| {
				let [contents] = v.contents;
				contents
			}),
		}
	}
	
	/// Retrieve an array of column vectors from a matrix.
	///
	/// # Examples
	///
	#[doc(include = "../doc/transform.md")]
	pub fn to_vectors(self) -> [Vector<T, M>; N] {
		self.contents.map(|v| {
			Matrix::vector(v)
		})
	}
	
	/// Swap two rows of a matrix.
	pub fn swap_row(&mut self, i: usize, j: usize) {
		for k in 0..N {
			self.contents[k].swap(i, j);
		}
	}
	
	/// Swap two columns of a matrix.
	pub fn swap_col(&mut self, i: usize, j: usize) {
		self.contents.swap(i, j);
	}
	
	/// `Matrix`-`Matrix` left-division.
	///
	/// This performs a generalised Gauss-Jordan elimination to calculate `rhs^-1 * self`.
	///
	/// See [`Div<Matrix<TRhs, N, N>>`](#impl-Div<Matrix<TRhs%2C%20N%2C%20N>>).
	pub fn div_left<TRhs>(self, rhs: Matrix<TRhs, M, M>) -> Self where
		Matrix<T, N, M>: Div<Matrix<TRhs, M, M>, Output = Matrix<T, N, M>>
	{
		(self.transpose() / rhs.transpose()).transpose()
	}
}

impl<T, const M: usize, const N: usize> !Scalar for Matrix<T, M, N> { }

impl<T, const M: usize, const N: usize> Init<T, [usize; 2]> for Matrix<T, M, N> {
	fn init_with<F: FnMut([usize; 2]) -> T>(_: (), mut elem: F) -> Self {
		Self::new(<[_; N]>::init(|col| <[_; M]>::init(|row| elem([row, col]))))
	}
}

impl<T, const M: usize, const N: usize> Index<[usize; 2]> for Matrix<T, M, N> {
	type Output = T;
	
	fn index(&self, [row, col]: [usize; 2]) -> &T {
		&self.contents[col][row]
	}
}

impl<T, const M: usize, const N: usize> IndexMut<[usize; 2]> for Matrix<T, M, N> {
	fn index_mut(&mut self, [row, col]: [usize; 2]) -> &mut T {
		&mut self.contents[col][row]
	}
}

impl<T, const M: usize, const N: usize> Map for Matrix<T, M, N> {
	type TFrom = T;
	type TOut<TTo> = Matrix<TTo, M, N>;
	
	fn map<TTo, F: FnMut(Self::TFrom) -> TTo>(self, mut f: F) -> Self::TOut<TTo> {
		Matrix::new(self.contents.map(|col| col.map(|x| f(x))))
	}
}

impl<TLhs, const M: usize, const N: usize> Zip for Matrix<TLhs, M, N> {
	type TLhs = TLhs;
	type TSelf<T> = Matrix<T, M, N>;
	
	fn zip<TRhs, TTo, F: FnMut(Self::TLhs, TRhs) -> TTo>(self, rhs: Self::TSelf<TRhs>, mut f: F) -> Self::TSelf<TTo> {
		Matrix::new(self.contents.zip(rhs.contents, |a_col, b_col| a_col.zip(b_col, |a, b| f(a, b))))
	}
}

impl<T: Copy, const M: usize, const N: usize, const M_OUT: usize, const N_OUT: usize> Section<[usize; 2], Matrix<T, M_OUT, N_OUT>> for Matrix<T, M, N> {
	fn section(&self, [row_offset, col_offset]: [usize; 2]) -> Matrix<T, M_OUT, N_OUT> {
		assert!(row_offset <= M - M_OUT && col_offset <= N - N_OUT, "Out of bounds");
		Matrix::init(|[row, col]: [usize; 2]| self[[row + row_offset, col + col_offset]])
	}
}


impl<T: Copy, const M: usize, const N: usize> Zero for Matrix<T, M, N> where
	T: Zero,
{
	fn zero() -> Self {
		Self::init(|_| T::zero())
	}
	
	fn is_zero(&self) -> bool {
		for col in 0..N {
			for row in 0..M {
				if !self[[row, col]].is_zero() {
					return false;
				}
			}
		}
		true
	}
}

impl<T: Copy, const N: usize> One for Matrix<T, N, N> where
	T: Zero + One + MulAdd<Output = T>,
{
	fn one() -> Self {
		Self::init(|[row, col]| if row == col { T::one() } else { T::zero() })
	}
	
	// FIXME: Uncomment when `One` relaxes the `PartialEq` requirement
	
	/*
	fn is_one(&self) -> bool {
		for col in 0..N {
			for row in 0..N {
				if
					if col == row { self[[row, col]].is_one() }
					else { self[[row, col]].is_zero() }
				{
					return false
				}
			}
		}
		true
	}
	*/
}

// FIXME: Switch to derived traits once they're implemented for arbitrary length arrays

impl<T, const M: usize, const N: usize> PartialEq for Matrix<T, M, N> where
	T: PartialEq,
{
	fn eq(&self, rhs: &Matrix<T, M, N>) -> bool {
		for col in 0..N {
			for row in 0..M {
				if self[[row, col]] != rhs[[row, col]] {
					return false;
				}
			}
		}
		true
	}
}

impl<T, const M: usize, const N: usize> Eq for Matrix<T, M, N> where
	T: Eq,
{ }

impl<T, const M: usize, const N: usize> Hash for Matrix<T, M, N> where
	T: Hash,
{
	fn hash<H: Hasher>(&self, state: &mut H) {
		for col in 0..N {
			for row in 0..M {
				self[[row, col]].hash(state);
			}
		}
	}
}

impl<T, const M: usize, const N: usize> Debug for Matrix<T, M, N> where
	T: Debug,
{
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
		struct Cell<'a, T, const M: usize, const N: usize> { mat: &'a Matrix<T, M, N>, row: usize, col: usize }
		impl<T: Debug, const M: usize, const N: usize> Debug for Cell<'_, T, M, N> {
			fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
				self.mat[[self.row, self.col]].fmt(f)
			}
		}
		struct Row<'a, T, const M: usize, const N: usize> { mat: &'a Matrix<T, M, N>, row: usize }
		impl<T: Debug, const M: usize, const N: usize> Debug for Row<'_, T, M, N> {
			fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
				<[Cell<T, M, N>; N]>::init(|col| Cell { mat: self.mat, row: self.row, col })[..].fmt(f)
			}
		}
		f.debug_tuple("Matrix")
			.field(&&<[Row<T, M, N>; M]>::init(|row| Row { mat: self, row })[..])
			.finish()
	}
}

impl<T, const M: usize, const N: usize> Default for Matrix<T, M, N> where
	T: Default,
{
	fn default() -> Self {
		Self::init(|_| T::default())
	}
}

impl<T, const M: usize, const N: usize> Transpose for Matrix<T, M, N> {
	type Output = Matrix<T, N, M>;
	
	fn transpose(self) -> Self::Output {
		Matrix::new(self.contents.transpose())
	}
}

/// `Matrix`-`Scalar` multiplication.
impl<TLhs: Copy, TRhs: Copy, TOutput, const M: usize, const N: usize> Mul<TRhs> for Matrix<TLhs, M, N> where
	TLhs: Mul<TRhs, Output = TOutput>,
	TRhs: Scalar,
{
	type Output = Matrix<TOutput, M, N>;
	
	fn mul(self, rhs: TRhs) -> Self::Output {
		self.map(|x| x * rhs)
	}
}

/// `Matrix`-`Scalar` multiplication.
impl<TLhs: Copy, TRhs: Copy, const M: usize, const N: usize> MulAssign<TRhs> for Matrix<TLhs, M, N> where
	Self: Mul<TRhs, Output = Matrix<TLhs, M, N>>,
	TRhs: Scalar,
{
	fn mul_assign(&mut self, rhs: TRhs) {
		*self = *self * rhs;
	}
}

/// `Matrix`-`Scalar` division.
impl<TLhs: Copy, TRhs: Copy, TOutput, const M: usize, const N: usize> Div<TRhs> for Matrix<TLhs, M, N> where
	TLhs: Div<TRhs, Output = TOutput>,
	TRhs: Scalar,
{
	type Output = Matrix<TOutput, M, N>;
	
	fn div(self, rhs: TRhs) -> Self::Output {
		self.map(|x| x / rhs)
	}
}

/// `Matrix`-`Scalar` division.
impl<TLhs: Copy, TRhs: Copy, const M: usize, const N: usize> DivAssign<TRhs> for Matrix<TLhs, M, N> where
	Self: Div<TRhs, Output = Matrix<TLhs, M, N>>,
	TRhs: Scalar,
{
	fn div_assign(&mut self, rhs: TRhs) {
		*self = *self / rhs;
	}
}

impl<TLhs: Copy, TRhs: Copy, TOutput, const M: usize, const N: usize> Add<Matrix<TRhs, M, N>> for Matrix<TLhs, M, N> where
	TLhs: Add<TRhs, Output = TOutput>,
{
	type Output = Matrix<TOutput, M, N>;
	
	fn add(self, rhs: Matrix<TRhs, M, N>) -> Self::Output {
		self.zip(rhs, |a, b| a + b)
	}
}

impl<TLhs: Copy, TRhs: Copy, const M: usize, const N: usize> AddAssign<Matrix<TRhs, M, N>> for Matrix<TLhs, M, N> where
	Self: Add<Matrix<TRhs, M, N>, Output = Self>,
{
	fn add_assign(&mut self, rhs: Matrix<TRhs, M, N>) {
		*self = *self + rhs;
	}
}

impl<TLhs: Copy, TRhs: Copy, TOutput, const M: usize, const N: usize> Sub<Matrix<TRhs, M, N>> for Matrix<TLhs, M, N> where
	TLhs: Sub<TRhs, Output = TOutput>,
{
	type Output = Matrix<TOutput, M, N>;
	
	fn sub(self, rhs: Matrix<TRhs, M, N>) -> Self::Output {
		self.zip(rhs, |a, b| a - b)
	}
}

impl<TLhs: Copy, TRhs: Copy, const M: usize, const N: usize> SubAssign<Matrix<TRhs, M, N>> for Matrix<TLhs, M, N> where
	Self: Sub<Matrix<TRhs, M, N>, Output = Self>,
{
	fn sub_assign(&mut self, rhs: Matrix<TRhs, M, N>) {
		*self = *self - rhs;
	}
}

impl<T: Copy, TOutput, const M: usize, const N: usize> Neg for Matrix<T, M, N> where
	T: Neg<Output = TOutput>,
{
	type Output = Matrix<TOutput, M, N>;
	
	fn neg(self) -> Self::Output {
		self.map(|x| -x)
	}
}

/// `Matrix`-`Matrix` multiplication.
impl<TLhs: Copy, TRhs: Copy, TOutput, const M: usize, const K: usize, const N: usize> Mul<Matrix<TRhs, K, N>> for Matrix<TLhs, M, K> where
	TLhs: Mul<TRhs, Output = TOutput> + MulAdd<TRhs, TOutput, Output = TOutput>,
	TOutput: Zero,
{
	type Output = Matrix<TOutput, M, N>;
	
	fn mul(self, rhs: Matrix<TRhs, K, N>) -> Self::Output {
		Self::Output::init(|[row, col]| {
			let mut result = TOutput::zero();
			for k in 0..K {
				result = self.contents[k][row].mul_add(rhs.contents[col][k], result);
			}
			result
		})
	}
}

/// `Matrix`-`Matrix` multiplication.
impl<TLhs: Copy, TRhs: Copy, const M: usize, const N: usize> MulAssign<Matrix<TRhs, N, N>> for Matrix<TLhs, M, N> where
	Self: Mul<Matrix<TRhs, N, N>, Output = Self>,
{
	fn mul_assign(&mut self, rhs: Matrix<TRhs, N, N>) {
		*self = *self * rhs;
	}
}

impl<TLhs: Copy, TA: Copy, TB: Copy, const M: usize, const K: usize, const N: usize> MulAdd<Matrix<TA, K, N>, Matrix<TB, M, N>> for Matrix<TLhs, M, K> where
	TLhs: MulAdd<TA, TB, Output = TB>,
{
	type Output = Matrix<TB, M, N>;
	
	fn mul_add(self, a: Matrix<TA, K, N>, b: Matrix<TB, M, N>) -> Self::Output {
		Self::Output::init(|[row, col]| {
			let mut result = b[[row, col]];
			for k in 0..K {
				result = self.contents[k][row].mul_add(a.contents[col][k], result);
			}
			result
		})
	}
}

impl<TLhs: Copy, TA: Copy, TB: Copy, const M: usize, const N: usize> MulAddAssign<Matrix<TA, N, N>, Matrix<TB, M, N>> for Matrix<TLhs, M, N> where
	Self: MulAdd<Matrix<TA, N, N>, Matrix<TB, M, N>, Output = Self>,
{
	fn mul_add_assign(&mut self, a: Matrix<TA, N, N>, b: Matrix<TB, M, N>) {
		*self = self.mul_add(a, b);
	}
}

/// `Matrix`-`Matrix` right-division.
///
/// This performs a generalised Gauss-Jordan elimination to calculate `self * rhs^-1`.
///
/// This is slightly more efficient than calculating the inverse then multiplying, as the
/// Gauss-Jordan method of finding an inverse involves multiplying the identity matrix by the
/// inverse of the matrix, so you can replace this identity matrix with another matrix to get an
/// extra multiplication 'for free'. Inversion is then defined as `A^-1 = 1 / A`.
impl<TLhs: Copy, TRhs: Copy, const M: usize, const N: usize> Div<Matrix<TRhs, N, N>> for Matrix<TLhs, M, N> where
	TLhs: MulAdd<TRhs, TLhs, Output = TLhs> + DivAssign<TRhs>,
	TRhs: Zero + MulAdd<TRhs, TRhs, Output = TRhs> + DivAssign<TRhs> + Neg<Output = TRhs>,
{
	type Output = Self;
	
	fn div(mut self, mut rhs: Matrix<TRhs, N, N>) -> Self::Output {
		// The Gauss-Jordan method 'multiplies' `A|B` by `A^-1` to give `I|A^-1 B`
		
		// We want `A B^-1`, so we can apply transpositions to get this
		
		// `(A^T)^-1 = (A^-1)^T`
		// `(A B)^T = B^T A^T`
		// Therefore `A B^-1 = ((B^-1)^T A^T)^T = ((B^T)^-1 A^T)^T`
		// Applying Gauss-Jordan elimination to `A^T|B^T` gives `(B^T)^-1 A^T`
		// We can then transpose this to give `((B^T)^-1 A^T)^T = A B^-1`
		
		// To improve this further we can also remove the need for calculating any of the transpositions
		// by switching the ordering of the indexing, which ends up being slightly more efficient anyway
		// as it involves swapping columns instead of rows, which is often faster.
		
		for i in 0..N {
			if let Some(j) = (i..N).find(|j| !rhs[[*j, *j]].is_zero()) {
				if i != j {
					rhs.swap_col(i, j);
					self.swap_col(i, j);
				}
				
				let factor = rhs[[i, i]];
				// Never going to use this element again, so can skip this
				// rhs[[i, i]] = T::one();
				for k in i+1..N {
					rhs[[k, i]] /= factor;
				}
				for k in 0..M {
					self[[k, i]] /= factor;
				}
				
				for j in (0..i).chain(i+1..N) {
					let factor = -rhs[[i, j]];
					// Never going to use this element again, so can skip this
					// rhs[[i, j]] = T::zero();
					for k in i+1..N {
						rhs[[k, j]] = rhs[[k, i]].mul_add(factor, rhs[[k, j]]);
					}
					for k in 0..M {
						self[[k, j]] = self[[k, i]].mul_add(factor, self[[k, j]]);
					}
				}
			} else {
				panic!("Matrix has no inverse");
			}
		}
		
		self
	}
}

/// `Matrix`-`Matrix` division.
///
/// See [Div<Matrix<TRhs, N, N>>](struct.Matrix.html#impl-Div%3CMatrix%3CTRhs%2C%20N%2C%20N%3E%3E)
impl<T: Copy, const M: usize, const N: usize> DivAssign<Matrix<T, N, N>> for Matrix<T, M, N> where
	Self: Div<Matrix<T, N, N>, Output = Self>,
{
	fn div_assign(&mut self, rhs: Matrix<T, N, N>) {
		*self = *self / rhs;
	}
}

impl<T: Copy, const N: usize> Inv for Matrix<T, N, N> where
	Self: One + Div<Self, Output = Self>,
{
	type Output = Self;
	
	fn inv(self) -> Self::Output {
		Self::one().div(self)
	}
}

// FIXME: Remove the `Inv` bound for `Unsigned` once `Signed` and `Unsigned` are mutually exclusive
impl<T: Copy, TRhs, const N: usize> Pow<TRhs> for Matrix<T, N, N> where
	Self: Inv<Output = Self> + One + MulAssign<Self>,
	TRhs: PrimInt,
{
	type Output = Self;
	
	fn pow(mut self, mut rhs: TRhs) -> Self::Output {
		if rhs < TRhs::zero() {
			self = self.inv();
			rhs = TRhs::zero() - rhs;
		}
		
		let mut result = Self::one();
		
		while rhs > TRhs::zero() {
			if rhs & TRhs::one() == TRhs::one() {
				result *= self;
			}
			rhs = rhs >> 1;
			self *= self;
		}
		
		result
	}
}
