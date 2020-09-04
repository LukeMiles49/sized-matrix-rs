use sized_matrix::Vector;
use higher_order_functions::Init;

#[test]
fn init_vector() {
	let vector = Vector::<_, 4>::init(|x: usize| x);
	
	for i in 0..4 {
		assert_eq!(vector[i], i);
	}
}

#[test]
fn matrix_vector() {
	let vector = Vector::vector([0, 1, 2, 3]);
	
	assert_eq!(vector, Vector::<_, 4>::init(|x: usize| x));
}
