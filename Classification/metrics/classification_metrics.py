import torch 


def accuracy(outputs: list[torch.Tensor], labels: list[torch.Tensor]):
    length = sum(output.size(0) for output in outputs)
    correct = 0
    for i in range(len(outputs)):
        pred = outputs[i].argmax(dim=1, keepdim=True)
        correct+= pred.eq(labels[i].view_as(pred)).sum().item()
    return (correct/length)*100

