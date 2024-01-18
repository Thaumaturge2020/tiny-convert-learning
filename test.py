import torch


x = torch.ones(2, 2)
print(x)

x = torch.nn.functional.pad(x, (1, 1), mode='reflect')
print(x)


x = torch.tensor([[1.,2.],[3.,4.]])

x = torch.nn.functional.pad(x,(0,2),mode='circular')

x = torch.tensor([1.,2.,3.,4.])

y = torch.tensor([2.,1.,4.,3.])

x = (x > y)

z = torch.tensor([0.8,0.2,0.3,0.6])

y = (z > 0.5)


print(x,y)

x = torch.where(x,1.0,0.0)

z = torch.logical_xor(x,y)

w = torch.logical_not(z)

print(w,z)