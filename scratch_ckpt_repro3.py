import torch, traceback
from torch.utils.checkpoint import checkpoint
from omnivggt.layers.block import Block
from omnivggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
torch.manual_seed(0)
dev,DIM,HEADS,B,H,W="cuda",1024,16,2,39,31
rope=RotaryPositionEmbedding2D(frequency=100).to(dev); posget=PositionGetter(); P=H*W
blks=[Block(dim=DIM,num_heads=HEADS,qk_norm=True,rope=rope).to(dev).train() for _ in range(2)]
for b in blks: b.requires_grad_(True)
x0=torch.randn(B,P,DIM,device=dev,dtype=torch.float32); pos=posget(B,H,W,dev)
try:
    with torch.autocast("cuda",dtype=torch.bfloat16,cache_enabled=True):
        with torch.no_grad(): _=blks[0](x0,pos=pos)   # warm cache
        x=x0
        for b in blks: x=checkpoint(lambda inp,p,bb=b: bb(inp,pos=p),x,pos,use_reentrant=False)
        x.float().sum().backward()
except Exception as e:
    print(str(e)[:1500])
