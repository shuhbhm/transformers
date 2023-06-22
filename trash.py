from nn import MLP


# data
xs = [
    [2.0,3.0,-1.0],
    [3.0,-1.0,0.5],
    [0.5,1.0,1.0],
    [1.0,1.0,-1.0]
]

# label
ys = [1.0,-1.0,-1.0,1.0] # desired targets





y_pred = list()


n = MLP(3,[4,4,1])


for k in range(200):
    # forward pass
    y_pred = [n(x) for x in xs]
    loss = sum([(ygt - yout)**2 for ygt , yout in zip(ys,y_pred)])
    
    # backward pass
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()
    
    #  update weights
    for p in n.parameters():
        p.data += -0.01 * p.grad    
        
    print(k,'\t',loss.data)
    
    

print('-'*50)

for i in y_pred:
    print(i.data)


print('-'*50)


