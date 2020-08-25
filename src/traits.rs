use num_traits::Num;

/// Types which are used to scale [Matrices](struct.Matrix.html).
pub trait Scalar { }
impl<T: Num> Scalar for T { }

/// Unary operator for transposing a value.
pub trait Transpose {
	type Output;
	
	fn transpose(self) -> Self::Output;
}
