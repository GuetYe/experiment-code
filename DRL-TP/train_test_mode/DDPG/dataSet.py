from torch.utils.data import Dataset,DataLoader

class MyDataSets(Dataset):
    def __init__(self,x,y):
        super(MyDataSets,self).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index],self.y[index]

    def __len__(self):
        return len(self.x)

class dataLoader:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def load_set(self, train_data, target_data):
        train_data = MyDataSets(train_data, target_data)
        train_loader = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=False)
        return train_loader
