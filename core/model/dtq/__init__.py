from .dtq import DTQ
from .util import Dataset

def train(train_img, database_img, query_img, config):
    model = DTQ(config)
    img_database = Dataset(database_img, config.output_dim, config.subspace * config.subcenter)
    img_query = Dataset(query_img, config.output_dim, config.subspace * config.subcenter)
    img_train = Dataset(train_img, config.output_dim, config.subspace * config.subcenter)
    model.train_cq(img_train, img_query, img_database, config.R)
    return model.save_dir


def validation(database_img, query_img, config):
    model = DTQ(config)
    img_database = Dataset(database_img, config.output_dim, config.subspace * config.subcenter)
    img_query = Dataset(query_img, config.output_dim, config.subspace * config.subcenter)
    return model.validation(img_query, img_database, config.R)
