# encoding: utf-8
"""
"""

import logging
import sys

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer
from fastreid.modeling import build_model
from fastreid.engine import default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer

from fastreattribute.build_data import build_reattribute_train_loader, build_reattribute_test_loader
from fastreattribute.config import add_reattribute_config
from fastreattribute.reattribute_evaluator import ReAttribuetEvaluator
from fastreid.evaluation import ReidEvaluator

class Trainer(DefaultTrainer):

    def build_train_loader(self, cfg):
        data_loader = build_reattribute_train_loader(cfg)
        return data_loader

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        test_loader, num_query = build_reattribute_test_loader(cfg, dataset_name)
        return test_loader, num_query

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        if cfg.TEST.ATTRIBUTE.ENABLED:
            data_loader, num_query = cls.build_test_loader(cfg, dataset_name)
            return data_loader, ReAttribuetEvaluator(cfg)
        else:
            data_loader, num_query = cls.build_test_loader(cfg, dataset_name)
            return data_loader, ReidEvaluator(cfg, num_query, None)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_reattribute_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = Trainer.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
