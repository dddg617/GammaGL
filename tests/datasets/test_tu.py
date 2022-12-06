
def test_ENZYMES(get_dataset):
    dataset = get_dataset(name='ENZYMES')
    assert len(dataset) == 600
    assert dataset.num_classes == 6
    assert dataset.num_node_features == 3

def test_entities(get_dataset):
	dataset = get_dataset(name = 'MUTAG')
	g = dataset[0]

	print(len(dataset))
	print(dataset.num_features)
	print(dataset.num_classes)

	assert len(dataset) == 188

	assert dataset.num_features == 7
	assert dataset.num_classes == 2
	assert str(dataset) == 'MUTAG(188)'
	print(dataset[0])
	assert len(dataset[0]) == 3
 