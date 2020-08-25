use super::Transpose;

use core::mem::{MaybeUninit, transmute_copy, forget, replace};

// FIXME: Replace with some other library that does this once they start supporting const generics

pub trait MapArray<T, const N: usize> {
	fn map<U, F: FnMut(T) -> U>(self, f: F) -> [U; N];
}

impl<T, const N: usize> MapArray<T, N> for [T; N] {
	fn map<U, F: FnMut(T) -> U>(self, mut elem: F) -> [U; N] {
		let mut consumed: [MaybeUninit<T>; N] = unsafe { transmute_copy(&self) };
		forget(self);
		let mut contents: [MaybeUninit<U>; N] = MaybeUninit::uninit_array();
		
		for i in 0..N {
			contents[i] = MaybeUninit::new(elem(unsafe { replace(&mut consumed[i], MaybeUninit::uninit()).assume_init() }));
		}
		
		let res: [U; N] = unsafe { transmute_copy(&contents) };
		forget(contents);
		res
	}
}

impl<T, const M: usize, const N: usize> Transpose for [[T; M]; N] {
	type Output = [[T; N]; M];
	
	fn transpose(self) -> [[T; N]; M] {
		let mut consumed: [[MaybeUninit<T>; M]; N] = unsafe { transmute_copy(&self) };
		forget(self);
		let mut contents: [[MaybeUninit<T>; N]; M] = unsafe { MaybeUninit::uninit().assume_init() };
		
		for i in 0..M {
			for j in 0..N {
				contents[i][j] = replace(&mut consumed[j][i], MaybeUninit::uninit());
			}
		}
		
		let res: [[T; N]; M] = unsafe { transmute_copy(&contents) };
		forget(contents);
		res
	}
}
