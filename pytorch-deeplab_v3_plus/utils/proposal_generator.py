import multiprocessing as mp
import tempfile, shutil, os
import io, pickle
import torch
import torch.nn.functional as F
import gzip


class AlphaBasedProposalGenerator(object):
    def __init__(self, alpha_expansion, eps=0):
        self.alpha_expansion = alpha_expansion
        self.model = None
        self.eps = eps


    def get_unary(self, logits):
        if self.eps == 0:
            return -F.log_softmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        C = logits.shape[1]
        return -torch.log((1 - C * self.eps / (C - 1)) * probs + self.eps / (C - 1))


    def get_model(self):
        # return self.model
        if self.cached_model is None:
            self.cached_model = torch.load(io.BytesIO(self.model), "cpu")
            self.cached_model.cpu()
            self.cached_model.eval()
        return self.cached_model


    def update_model(self, model, read_cache):
        self.read_cache = read_cache
        self.cached_model = None
        f = io.BytesIO()
        torch.save(model, f)
        self.model = f.getvalue()


    def __call__(self, image, target, index):
        if self.read_cache:
            return self.load(index)
        x0  = None
        if self.read_cache is not None:
            x0 = self.load(index)[0]
            x0.unsqueeze_(0)
        image.unsqueeze_(0)
        target.unsqueeze_(0)
        croppings = (target != 254).float()

        sz = image.shape[-2:]
        with torch.no_grad():
            logits = self.get_model()(image)
        result = self.alpha_expansion(
            self.get_unary(logits), image, croppings, target, x0=x0, index=index)
        result[0].squeeze_(0)
        self.save(result, index)
        return result



class ProposalGeneratorSharedMem(AlphaBasedProposalGenerator):
    def __init__(self, alpha_expansion, eps=0):
        super().__init__(alpha_expansion, eps)
        self.model = None
        self.read_cache = False
        self.manager = mp.Manager()
        self.hidden_label_cache = self.manager.dict()


    def load(self, index):
        return self.hidden_label_cache[index]


    def save(self, obj, index):
        self.hidden_label_cache[index] = obj



class ProposalGeneratorFileCache(AlphaBasedProposalGenerator):
    def __init__(self, alpha_expansion, eps=0, path=None, del_path=None):
        super().__init__(alpha_expansion, eps)
        self.cached_model = None
        self.read_cache = None
        self.del_path = del_path or path is None
        self.path = path or tempfile.mkdtemp(dir=os.environ.get("SLURM_TMPDIR"))
        print("Saving proposals in %s" % self.path)


    def __del__(self):
        if self.del_path:
            print("Deleting temp directory %s" % self.path)
            shutil.rmtree(self.path)


    def file_name(self, index):
        return '%s/%05d.pt' % (self.path, index)


    def load(self, index):
        return torch.load(gzip.open(self.file_name(index)))


    def save(self, obj, index):
        return torch.save(obj, gzip.open(self.file_name(index), 'w'))
