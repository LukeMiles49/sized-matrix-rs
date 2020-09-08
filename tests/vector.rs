use sized_matrix::{Vector, Dot};
use higher_order_functions::{Init, Section};

#[test]
fn init_vector() {
	let vector = Vector::<_, 4>::init(|x: usize| x);
	
	for i in 0..4 {
		assert_eq!(vector[i], i);
	}
}

#[test]
fn vector_vector() {
	let vector = Vector::vector([0, 1, 2, 3]);
	
	assert_eq!(vector, Vector::<_, 4>::init(|x: usize| x));
}

#[test]
fn vector_section() {
	let vector = Vector::vector([0, 1, 2, 3]);
	
	assert_eq!(vector.section(1), Vector::vector([1, 2]));
}

#[test]
fn vector_dot() {
	let a = Vector::vector([0, 1, 1, 3, 5]);
	
	let b = Vector::vector([2, 3, 5, 7, 11]);
	
	assert_eq!(a.dot(b), 84);
}
