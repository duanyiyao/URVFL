import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

def get_cos_sim(v1, v2):
  num = float(np.dot(v1,v2))
  denom = np.linalg.norm(v1) * np.linalg.norm(v2)
  # return num / denom if denom != 0 else 0
  return 0.5 + 0.5 * (num / denom) if denom != 0 else 0

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

# claculate detection score
def detection_score_compute(similarity, step):


    poly_n = 2

    ## normal training
  #  similarity = similarity[:, :length]

    n_len = similarity.shape[1]
    x = np.arange(n_len)
    x2 = np.arange(50, n_len, step)

    z1 = np.polyfit(x, similarity[0], poly_n)
    p1 = np.poly1d(z1)
    pp1 = p1(x2)
   
    z2 = np.polyfit(x, similarity[1], poly_n)
    p2 = np.poly1d(z2)
    pp2 = p2(x2)
  
    same_variance_min = similarity[1][50::step] - similarity[3][50::step]
    same_variance_max = similarity[1][50::step] + similarity[3][50::step]
    diff_variance_min = similarity[0][50::step] - similarity[2][50::step]
    diff_variance_max = similarity[0][50::step] + similarity[2][50::step]
    

    
   # iou_in = np.where(np.maximum(same_variance_min, diff_variance_min)> np.minimum(diff_variance_max, same_variance_max),
   #                   np.zeros_like(same_variance_min),np.minimum(diff_variance_max, same_variance_max)-np.minimum(same_variance_min, diff_variance_min))
   # iou_range = np.maximum(diff_variance_max, same_variance_max)-np.minimum(same_variance_min, diff_variance_min)
   # iou_score = iou_in/iou_range
   
   
    ##new
    epsilon = 1e-8
    iou_score = []
    for k in range(len(same_variance_min)):
        if same_variance_min[k] <= diff_variance_max[k]:
            over_ratio_a = diff_variance_max[k] - same_variance_min[k]
            over_ratio_b = same_variance_max[k] - diff_variance_min[k]
            iou_score.append(over_ratio_a / (over_ratio_b + epsilon))
        else:
            iou_score.append(0.0)
    iou_score = np.array(iou_score)
   
 
    gap_score = (similarity[1, :] - similarity[0, :]) / (similarity[0, :] + similarity[1, :])

    err_score = []
    for i in range(50, n_len, step):
        x_current = np.arange(i)
        z_current = np.polyfit(x_current, similarity[0][:i], poly_n)
        p1_current = np.poly1d(z_current)
        pp1_current = p1_current(x_current)
        err = np.sqrt(np.mean((similarity[0][:i]-pp1_current)**2))
        err_score.append(err)
    err_score = np.array(err_score)


    err_diff_score = []
    for i in range(50, n_len, step):
        x_current = np.arange(i)
        z_current = np.polyfit(x_current, similarity[1][:i], poly_n)
        p1_current = np.poly1d(z_current)
        pp1_current = p1_current(x_current)
        err = np.sqrt(np.mean((similarity[1][:i] - pp1_current) ** 2))
        err_diff_score.append(err)
    err_diff_score = np.array(err_diff_score)
    
    
  


    epsilo = 1/(np.e*np.e*np.e)
    la = 9
    la_g =  0.8
    la_ds = 6
    la_iou = 1
    eposilo_iou = 1/np.e

 
    modified_err = (err_score + err_diff_score) / 2 * la
    m_err = -np.log(modified_err + epsilo)
    m_iou = -np.log(iou_score*la_iou + eposilo_iou)
    m_gap = gap_score[50::step]/(np.abs(gap_score[50::step])**la_g)
    final_error = m_gap * m_err * m_iou -0.7
    final_error = sigmoid(la_ds *final_error)

    return final_error

def angle(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(dot_product)

def sigmoid(x, shift=0, mult=1, exp=1):
    x_p = (x - shift) * mult
    return (1 / (1 + np.exp(-x_p))) ** exp

def sg_score(fakes, controls, regulars, shift=0, mult=1, exp=1, raw=False):
    f_mean = sum(fakes) / len(fakes)
    c_mean = sum(controls) / len(controls)
    r_mean = sum(regulars) / len(regulars)
    cr_mean = (c_mean + r_mean) / 2

    f_mean_mag = sum([np.linalg.norm(v) for v in fakes]) / len(fakes)
    c_mean_mag = sum([np.linalg.norm(v) for v in controls]) / len(controls)
    r_mean_mag = sum([np.linalg.norm(v) for v in regulars]) / len(regulars)
    cr_mean_mag = (c_mean_mag + r_mean_mag) / 2

    mag_div = (abs(f_mean_mag - cr_mean_mag) + abs(c_mean_mag - r_mean_mag))

    x = angle(f_mean, cr_mean) * (abs(f_mean_mag - cr_mean_mag) / mag_div) - angle(c_mean, r_mean) * (abs(r_mean_mag - c_mean_mag) / mag_div)

    if raw:
        return x
    else:
        return sigmoid(x, shift=shift, mult=mult, exp=exp)   

    


def efficient_cosine_similarities(gradients, labels):
    # Normalize the gradients to have unit norm
    norms = torch.norm(gradients, p=2, dim=1, keepdim=True)
    normalized_gradients = gradients / norms
    
    # Compute all cosine similarities (dot product of normalized vectors)
    cosine_sim_matrix = torch.mm(normalized_gradients, normalized_gradients.t())
    
    # Create a label match matrix
    label_match_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
    
    # Extract similarities for same and different labels
    same_label_similarities = cosine_sim_matrix[label_match_matrix].tolist()
    different_label_similarities = cosine_sim_matrix[~label_match_matrix].tolist()
    
    return same_label_similarities, different_label_similarities


def get_grad_set(batch_grad, batch_label, dif_category_mean, dif_variance, same_category_mean, same_variance):
    same_category = []
    dif_category = []
    batch_size = batch_grad.size(0)
    batch_grad = batch_grad.reshape(batch_grad.size(0), -1)
    same_category, dif_category =  efficient_cosine_similarities(batch_grad, batch_label)
    
    # Compute mean and variance for different and same categories
    if dif_category:
        dif_category_mean_ = np.mean(dif_category)
        dif_variance_ = np.std(dif_category)
        dif_category_mean.append(dif_category_mean_)
        dif_variance.append(dif_variance_)

    if same_category:
        same_category_mean_ = np.mean(same_category)
        same_variance_ = np.std(same_category)
        same_category_mean.append(same_category_mean_)
        same_variance.append(same_variance_)

