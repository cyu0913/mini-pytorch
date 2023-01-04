class Value:
    '''
    Stores value and gradient of parameter.
    The object of Value type, can be considered a weight parameter in
    neural network. It has value, and gradient. Gradient can be computed 
    and accumulated depending on how it's connected and be contributed
    to the final weights. The Value has also contains operators, and these
    operators also contain both a foward and backward computation. 
    Forward computation computes the forward path, and backward computation
    propagate gradient backward according to chain rule dx/dz = dx/dy * dy/dz
    '''
    
    # _children will be mainly used for topological sort
    # during backward gradient computation. tuple () is used as 
    # the _children will not be modified. 
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def _match_type(self, data):
        return data if isinstance(data, Value) else Value(data)

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = self._match_type(other)
        output = Value(self.data + other.data, (self, other), '+')
    
        # when computing gradient backward, we could assume
        # that output.grad has computed alread
        def _backward():
            self.grad += 1 * output.grad
            other.grad += 1 * output.grad
        output._backward = _backward

        return output

    def __mul__(self, other):
        other = self._match_type(other)
        output = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * output.grad 
            other.grad += self.data * output.grad
        output._backward = _backward

        return output

    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        output = Value(self.data ** other, (self,), '**')

        def _backward():
            self.grad += (other * self.data**(other-1)) * output.grad
        output._backward = _backward

        return output  

    def __truediv__(self, other):
        other = self._match_type(other)
        return self * (other ** -1)

    # this handle the case 3 * a insteand of a * 3
    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return self - other

    def __rtruediv__(self, other):
        return other * (self ** -1)

    def __neg__(self):
        return self * -1

    def relu(self):
        output = Value(max(self.data, 0), (self,), 'ReLU')
        def _backward():
            self.grad += output.grad if self.data > 0.0 else 0.0
        output._backward = _backward
        return output

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


    
