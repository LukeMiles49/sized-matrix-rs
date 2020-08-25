//! Sized matrices using const generics for better type checking and performance.
//!
//! ```rust
//! use sized_matrix::Matrix;
//!
//! let a: Matrix<i32, 3, 4> = Matrix::rows([
//!     [ 1,  2,  3,  4],
//!     [ 5,  6,  7,  8],
//!     [ 9, 10, 11, 12],
//! ]);
//!
//! let b: Matrix<i32, 4, 2> = Matrix::rows([
//!     [ 0,  1],
//!     [ 1,  2],
//!     [ 3,  5],
//!     [ 8, 13],
//! ]);
//!
//! let c: Matrix<i32, 3, 2> = a * b;
//!
//! assert_eq!(c, Matrix::rows([
//!     [ 43,  72],
//!     [ 91, 156],
//!     [139, 240],
//! ]));
//! ```

#![no_std]

#![feature(const_generics)]
#![feature(external_doc)]
#![feature(maybe_uninit_uninit_array)]
#![feature(negative_impls)]

mod traits;
pub use traits::*;

mod matrix;
pub use matrix::*;

mod vector;
pub use vector::*;

mod array_helpers;
use array_helpers::*;

// Include the readme and changelog as hidden documentation so they're tested by cargo test
#[doc(include = "../README.md")]
#[doc(include = "../CHANGELOG.md")]
type _Doctest = ();
