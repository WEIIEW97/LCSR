from typing import List, Tuple
import os
from os.path import isdir, join, split
import torch
from pad2d_op import pad2d,pad2dT
from pad2d_op import Pad2d,Pad2dT

class TorchScriptModelUsingCustomOp(torch.nn.Module):
    def __init__(self, padding: List[int]=[1,1,1,1], mode:str="symmetric"):
        super(TorchScriptModelUsingCustomOp, self).__init__()
        self.padding: List[int] = padding
        self.mode: str = mode

    def forward(self, x):
        # Her any TorchScript compatible operations can be made (most of pytorch)
        x = x + 0.5 
        # ....
        x_pad = pad2d(x, self.padding, self.mode)
        return x, x_pad


if __name__ == "__main__":
    # Just making sure we find the correct folder for the C++ Application
    out_path = join( split(__file__)[0],"build")
    if not isdir(out_path):
        os.makedirs(out_path)

    # Export Model as TorchScript
    module = TorchScriptModelUsingCustomOp()
    jit_mod = torch.jit.script(module)
    jit_mod.save( join(out_path, "torch_scrip_model_using_costume_op.pt") )

    # Run the Demo Model to see what output should look like:
    data = torch.arange(9).reshape(3,3).float().cuda()
    out, out_padded = module(data)
    print("Demo Output (python)")
    print(out)
    print(out_padded)
