use num_traits::Num;

use core::mem::{MaybeUninit, transmute_copy, forget};

/// Types which are used to scale Matrices.
pub trait Scalar { }
impl<T: Num> Scalar for T { }

/// Unary operator for transposing a value.
pub trait Transpose {
	type Output;
	
	/// Returns the transpose of `self`.
	fn transpose(self) -> Self::Output;
}

impl<T, const M: usize, const N: usize> Transpose for [[T; M]; N] {
	type Output = [[T; N]; M];
	
	fn transpose(self) -> [[T; N]; M] {
		let consumed: [[MaybeUninit<T>; M]; N] = unsafe { transmute_copy(&self) };
		forget(self);
		let mut contents: [[MaybeUninit<T>; N]; M] = unsafe { MaybeUninit::uninit().assume_init() };
		
		for i in 0..M {
			for j in 0..N {
				contents[i][j].write(unsafe { consumed[j][i].assume_init_read() });
			}
		}
		
		unsafe { transmute_copy(&contents) }
	}
}

/// Binary operator for the dot product operation.
pub trait Dot<TRhs = Self> {
	type Output;
	
	/// Returns `self · rhs`.
	fn dot(self, rhs: TRhs) -> Self::Output;
}
