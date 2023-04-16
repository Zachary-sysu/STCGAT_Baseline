import numpy as np


def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(null_val)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')


# def masked_mape_np(y_true, y_pred, null_val=np.nan):
#     with np.errstate(divide='ignore', invalid='ignore'):
#         mask = mask_np(y_true, null_val)
#         mask /= mask.mean()
#         mape = np.abs((y_pred - y_true) / y_true)
#         mape = np.nan_to_num(mask * mape)
#         return np.mean(mape) * 100

# def masked_mape_np(y_true, y_pred, null_val=np.nan):
#     with np.errstate(divide='ignore', invalid='ignore'):
#         if np.isnan(null_val):
#             mask = ~np.isnan(y_true)
#         else:
#             mask = np.not_equal(y_true, null_val)
#         mask = mask.astype('float32')
#         mask /= np.mean(mask)
#         mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
#                       y_true))
#         mape = np.nan_to_num(mask * mape)
#         return np.mean(mape)


# def masked_mape_np(y_true, y_pred, null_val=np.nan):
#     with np.errstate(divide='ignore', invalid='ignore'):
#         if np.isnan(null_val):
#             mask = ~np.isnan(y_true)
#         else:
#             mask = np.not_equal(y_true, null_val)
#         mask = mask.astype('float32')
#         mask /= np.mean(mask)
#         mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
#                       y_true))
#         mape = np.nan_to_num(mask * mape)
#         return np.mean(mape)

def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)

# def masked_mape_np(preds, labels, null_val=np.nan):
#     with np.errstate(divide='ignore', invalid='ignore'):
#         if np.isnan(null_val):
#             mask = ~np.isnan(labels)
#         else:
#             mask = np.not_equal(labels, null_val)
#         mask = mask.astype('float32')
#         mask /= np.mean(mask)
#         mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
#         mape = np.nan_to_num(mask * mape)
#         return np.mean(mape)


def masked_rmse_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return np.sqrt(np.mean(np.nan_to_num(mask * mse)))


def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = np.abs(y_true - y_pred)
    return np.mean(np.nan_to_num(mask * mae))
