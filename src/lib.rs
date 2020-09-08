//! Sized matrices using const generics for better type checking and performance.
//!
//! ```rust
//! use sized_matrix::{Matrix, Vector};
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
//!
//! let d: Vector<i32, 2> = Matrix::vector([-1, 1]);
//!
//! let e: Vector<i32, 3> = c * d;
//!
//! assert_eq!(e, Matrix::vector([
//!      29,
//!      65,
//!     101,
//! ]));
//! ```
//!
//! To use this, add it as a dependency to your Cargo.toml:
//! ```toml
//! [dependencies]
//! sized_matrix = "^0.2.2"
//! ```

#![no_std]

#![feature(const_generics)]
#![feature(generic_associated_types)]
#![feature(external_doc)]
#![feature(maybe_uninit_uninit_array)]
#![feature(negative_impls)]
#![feature(maybe_uninit_extra)]

#![doc(html_root_url = "https://docs.rs/sized_matrix/0.2.2")]

mod traits;
pub use traits::*;

mod matrix;
pub use matrix::*;

mod vector;
pub use vector::*;

// Include the readme and changelog as hidden documentation so they're tested by cargo test
#[doc(include = "../README.md")]
#[doc(include = "../CHANGELOG.md")]
type _Doctest = ();
