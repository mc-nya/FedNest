import yaml

class Logger():
    def __init__(self, filename) -> None:
        self.filename=filename
        self.logs={"clients": [], "train_loss": [],
            "test_loss": [], "test_acc": [], "train_acc": []}
    def logging(self,client_idx, acc_test, acc_train, loss_test, loss_train):
        self.logs["clients"].append(client_idx.tolist())

        self.logs['test_acc'].append(acc_test.item())
        self.logs["train_acc"].append(acc_train.item())
        self.logs["test_loss"].append(loss_test)
        self.logs["train_loss"].append(loss_train)
    def save(self):
        f = open(self.filename, mode="w+")
        yaml.dump(self.logs, f)
        f.close()