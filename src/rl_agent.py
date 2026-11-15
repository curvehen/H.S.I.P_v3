# src/rl_agent.py
import random, numpy as np
class RLAgent:
    def __init__(self, models):
        self.models=models; self.q={m:0.0 for m in models}; self.alpha=0.1; self.gamma=0.9
    def choose(self):
        if random.random()<0.1: return random.choice(self.models)
        return max(self.q, key=self.q.get)
    def update(self, model, reward):
        best=max(self.q.values()); self.q[model]+=self.alpha*(reward+self.gamma*best-self.q[model])
    def weights(self):
        vals=np.array([self.q[m] for m in self.models]); vals=vals-vals.max(); exp=np.exp(vals*2)
        return {m: float(exp[i]/exp.sum()) for i,m in enumerate(self.models)}
    def learn(self, maes, rounds=200):
        for m, mae in maes.items(): self.update(m, -mae*100)
        for _ in range(rounds):
            c=self.choose(); self.update(c, -maes[c]*100 + np.random.randn()*0.01)
        return self.weights()
