from vision.datasets.egohands import EgoHandsDataset

#train = EgoHandsDataset('../images', transform=None, target_transform=None, dataset_type="train", balance_data=False)
test = EgoHandsDataset('../images', transform=None, target_transform=None, dataset_type="test", balance_data=False)

#print(train)
print(test)
