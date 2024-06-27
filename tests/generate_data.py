from scipy.stats import genextreme, gumbel_r, genpareto
import pandas as pd


def generate_genextreme():
    (shape_true, loc_true, scale_true) = (0.1, 55, 125)
    y = genextreme.rvs(shape_true, loc=loc_true, scale=scale_true, size=5000)
    df = pd.DataFrame(y, columns=['y'])
    df.to_csv('genextreme.csv', index=False)

def generate_genextreme_small():
    (shape_true, loc_true, scale_true) = (0.1, 55, 125)
    y = genextreme.rvs(shape_true, loc=loc_true, scale=scale_true, size=1000)
    df = pd.DataFrame(y, columns=['y'])
    df.to_csv('genextreme_small.csv', index=False)

def generate_gumbel_r():
    (loc_true, scale_true) = (80, 30)
    y = gumbel_r.rvs(loc=loc_true, scale=scale_true, size=5000)
    df = pd.DataFrame(y, columns=['y'])
    df.to_csv('gumbel_r.csv', index=False)

def generate_gumbel_r_small():
    (loc_true, scale_true) = (80, 30)
    y = gumbel_r.rvs(loc=loc_true, scale=scale_true, size=1000)
    df = pd.DataFrame(y, columns=['y'])
    df.to_csv('gumbel_r_small.csv', index=False)

def generate_pareto():
    (shape_true, loc_true, scale_true) = (0.1, 55, 125)
    y = genpareto.rvs(shape_true, loc=loc_true, scale=scale_true, size=5000)
    df = pd.DataFrame(y, columns=['y'])
    df.to_csv('genpareto.csv', index=False)

def generate_pareto_small():
    (shape_true, loc_true, scale_true) = (0.1, 55, 125)
    y = genpareto.rvs(shape_true, loc=loc_true, scale=scale_true, size=1000)
    df = pd.DataFrame(y, columns=['y'])
    df.to_csv('genpareto_small.csv', index=False)


# generate_genextreme()
# generate_genextreme_small()
# generate_gumbel_r()
# generate_gumbel_r_small()

generate_pareto()
generate_pareto_small()
