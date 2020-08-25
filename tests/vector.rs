use sized_matrix::Vector;
use init_trait::Init;

#[test]
fn init_vector() {
	let vector = Vector::<usize, 4>::init(|x: usize| x);
	
	for i in 0..4 {
		assert_eq!(vector[i], i);
	}
}

#[test]
fn matrix_vector() {
	let vector = Vector::vector([0, 1, 2, 3]);
	
	assert_eq!(vector, Vector::<usize, 4>::init(|x: usize| x));
}
