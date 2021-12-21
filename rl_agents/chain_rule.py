import torch


wa = torch.tensor([0.1], requires_grad=True)
ba = torch.tensor([0.2], requires_grad=True)
wq = torch.tensor([0.3, 0.4], requires_grad=True)
bq = torch.tensor([0.5], requires_grad=True)
s = torch.tensor([0.7, 0.8])

optimizer = torch.optim.Adam([wa, ba])
optimizer.zero_grad()

a = wa * s + ba
q = wq[0] * s + wq[1] * a + bq

loss = -q.mean()

# loss.backward(inputs=[wa, ba])
# print(wa.grad, ba.grad)
# print(a)
# OR
# a.backward(torch.ones_like(a), inputs=[wa,ba])
# print(wa.grad, ba.grad)
# loss.backward(inputs=a)
# print(a.grad)
# wa.grad *= a.grad
# ba.grad *= a.grad
# print(wa.grad, ba.grad)
# OR
loss.backward(inputs=[a])
a.backward(a.grad, inputs=[wa, ba])

optimizer.step()
print(wa.grad, ba.grad)
print(wa, ba)
print(a)

# a.backward(retain_graph=True)
# print(wa.grad, ba.grad)
# q.backward(inputs=a, retain_graph=True)
# print(a.grad)
# wa.grad.data.zero_()
# ba.grad.data.zero_()
# q.backward(inputs=[wa, ba])
# print(wa.grad, ba.grad)
