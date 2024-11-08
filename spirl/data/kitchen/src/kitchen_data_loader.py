import d4rl
import gym
import numpy as np
import itertools

from spirl.components.data_loader import Dataset
from spirl.utils.general_utils import AttrDict
from transformers import AutoTokenizer, PretrainedConfig, Dinov2Model, AutoImageProcessor, CLIPTextModel
from PIL import Image


class D4RLSequenceSplitDataset(Dataset):
    SPLIT = AttrDict(train=0.99, val=0.01, test=0.0)

    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1):
        self.phase = phase
        self.data_dir = data_dir
        self.spec = data_conf.dataset_spec
        self.subseq_len = self.spec.subseq_len
        self.remove_goal = self.spec.remove_goal if 'remove_goal' in self.spec else False
        self.dataset_size = dataset_size
        self.device = data_conf.device
        self.n_worker = 4
        self.shuffle = shuffle

        env = gym.make(self.spec.env_name)
        self.dataset = env.get_dataset()
        self.visual_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")


        # split dataset into sequences
        seq_end_idxs = np.where(self.dataset['terminals'])[0]
        start = 0
        self.seqs = []
        for end_idx in seq_end_idxs:
            if end_idx+1 - start < self.subseq_len: continue    # skip too short demos
            self.seqs.append(AttrDict(
                states=self.dataset['observations'][start:end_idx+1],
                actions=self.dataset['actions'][start:end_idx+1],
            ))
            start = end_idx+1

            
        # rgb
        self.image_dir = "data"
        # start = 0
        # for i, end_idx in enumerate(seq_end_idxs):
        #     if end_idx+1 - start < self.subseq_len: continue    # skip too short demos
        #     frames = []
        #     for idx in range(start, end_idx+1):
        #         env.set_state(self.dataset['observations'][idx, :30], np.zeros(29))
        #         img = env.render(mode='rgb_array')
        #         frames.append(img)
        #     frames = np.stack(frames)
        #     print("Start: {}, End: {}, Frames: {}".format(start, end_idx, frames.shape[0]))
        #     # np.save(f'{image_dir}/{str(i).zfill(4)}.npy', frames)
        #     start = end_idx+1

        # 0-pad sequences for skill-conditioned training
        if 'pad_n_steps' in self.spec and self.spec.pad_n_steps > 0:
            for seq in self.seqs:
                seq.states = np.concatenate((np.zeros((self.spec.pad_n_steps, seq.states.shape[1]), dtype=seq.states.dtype), seq.states))
                seq.actions = np.concatenate((np.zeros((self.spec.pad_n_steps, seq.actions.shape[1]), dtype=seq.actions.dtype), seq.actions))

        # filter demonstration sequences
        if 'filter_indices' in self.spec:
            print("!!! Filtering kitchen demos in range {} !!!".format(self.spec.filter_indices))
            if not isinstance(self.spec.filter_indices[0], list):
                self.spec.filter_indices = [self.spec.filter_indices]
            self.seqs = list(itertools.chain.from_iterable([\
                list(itertools.chain.from_iterable(itertools.repeat(x, self.spec.demo_repeats)
                               for x in self.seqs[fi[0] : fi[1]+1])) for fi in self.spec.filter_indices]))
            import random
            random.shuffle(self.seqs)

        self.n_seqs = len(self.seqs)

        if self.phase == "train":
            self.start = 0
            self.end = int(self.SPLIT.train * self.n_seqs)
        elif self.phase == "val":
            self.start = int(self.SPLIT.train * self.n_seqs)
            self.end = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
        else:
            self.start = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
            self.end = self.n_seqs

    def __getitem__(self, index):
        # sample start index in data range
        seq_idx = self._sample_seq()
        seq = self.seqs[seq_idx]
        start_idx = np.random.randint(0, seq.states.shape[0] - self.subseq_len - 1)
        frames = np.load(f'{self.image_dir}/{str(seq_idx).zfill(4)}.npy')
        
        prev_image = frames[start_idx]
        next_image = frames[start_idx+self.subseq_len]
        
        curr_features = np.stack(self.visual_processor(prev_image)['pixel_values'])
        next_features = np.stack(self.visual_processor(next_image)['pixel_values'])
        
        
        output = AttrDict(
            curr_features=curr_features,
            next_features=next_features,
            states=seq.states[start_idx:start_idx+self.subseq_len],
            actions=seq.actions[start_idx:start_idx+self.subseq_len-1],
            pad_mask=np.ones((self.subseq_len,)),
        )
        if self.remove_goal:
            output.states = output.states[..., :int(output.states.shape[-1]/2)]
        return output

    def _sample_seq(self):
        random_idx = np.random.randint(self.start, self.end)
        return random_idx

    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size
        return int(self.SPLIT[self.phase] * self.dataset['observations'].shape[0] / self.subseq_len)
