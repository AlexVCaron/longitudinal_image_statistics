#!/usr/bin/env python3

import itertools
from argparse import ArgumentParser
from functools import partial
from multiprocessing import cpu_count, Pool
from ntpath import join
import os
import re
import time
import warnings

import nibabel as nib
import numpy as np
from scipy.stats import bootstrap
from scipy.special import erf


"""
!!!IMPORTANT!!! All images must be in the same space and have identical 
dimensions in order for the script to run. Register everything together (or to 
a template) before usage.

Compute Pearson variation coefficients, Intra and Inter subjects variation 
coefficients and I2C2 intra-class correlation coefficients for dMRI metrics.
"""


def _create_parser():
    p = ArgumentParser(description=__doc__)
    p.add_argument("input",
                   help="Either a directory containing sub-directories of "
                        "measures or an HDF5 file. For the former, measure "
                        "names will be extracted from sub-directories names. "
                        "Each image file must be formatted as "
                        "<subject>_<session>___<...>.nii.gz.")
    p.add_argument("output_dir", help="Output directory")

    g1 = p.add_argument_group(
        "Masks", "Tissue mask(s) inside which all coefficients will "
                 "be computed. When supplying multiple masks, assert "
                 "that they follow the right naming convention :\n"
                 "  - A mask per subject : <subject>___<...>.nii.gz\n"
                 "  - A mask per subject per session : <subject>_<session>___"
                 "<...>.nii.gz\n"
    )
    meg = g1.add_mutually_exclusive_group()
    meg.add_argument("--tissue-masks",
                     nargs="+",
                     help="Either one mask can be passed for all subjects, a "
                          "mask per subject or a mask per subject per session. "
                          "See group description for the naming convention")
    meg.add_argument("--tissue-masks-dir",
                     help="Directory containing the masks. See the group "
                          "description for the masks naming convention")

    p.add_argument("--missing-data",
                   choices=["zero", "mean", "none", "exclude"],
                   default="mean",
                   help="Strategy used to fill missing datapoints "
                        "[%(default)s]. Either :\n "
                        "  (zeros) fill with zeros, this will bias statistical "
                        "measures\n"
                        "  (mean)  fill with non-missing datapoints average, "
                        "will reduce STD\n"
                        "  (none)  default behavior. Do nothing, which will "
                        "include the real \n"
                        "          values of missing data in averages and std "
                        "computation, but \n"
                        "          will exclude voxels with series missing "
                        "data from global \n"
                        "          averages")


    p.add_argument("--disable-corr-coeff", action="store_true")
    p.add_argument("--disable-var-coeff", action="store_true")
    p.add_argument("--disable-i2c2-coeff", action="store_true")

    g2 = p.add_argument_group(
        "Significance", "Measure significance of coefficients via "
                        "bootstrapping. Returns CI and p-values.")
    g2.add_argument("--significance",
                    action="store_true",
                    help="Enable computing the CI and p-values of "
                         "coefficients. This can be quite long on a big "
                         "dataset. Playing with multiprocessing parametres "
                         "can help.")
    g2.add_argument("--conf-interval",
                    type=float,
                    default=0.95,
                    help="Confidence interval to compute. [%(default)s]")
    g2.add_argument("--resample",
                    type=int,
                    default=100,
                    help="Number of bootstrap resample to execute. "
                         "[%(default)s]")
    g2.add_argument("--method",
                    choices=["percentile", "basic", "BCa"],
                    default="BCa",
                    help="Bootstrapping technique. For more information, see "
                         "the method parameter of scipy.stats.bootstrap. "
                         "[%(default)s]")
    g2.add_argument("--nb-threads",
                    type=int,
                    default=1,
                    help="Number of threads onto which the bootstrap is "
                         "parallelized. The amount of RAM required is equal "
                         "to the size of the data times the number of threads, "
                         "since the data will be copied on each. [%(default)s]")

    return p

# Computation on Masked array with Nan values

def _nanmean(data, axis=-1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nan_to_num(np.nanmean(data, axis=axis), copy=False)

# Single observation measures

def _i2c2(indexes, data, subjects, visits):
    _, _uix, _uct = np.unique(
        [subjects[i[0]] for i in indexes],
        return_index=True,
        return_counts=True)

    def trkw():
        return np.ma.sum(np.ma.sum((
            data[..., np.concatenate(
                indexes[_uix], dtype=int, casting='unsafe')]
            - _nanmean(data)[..., None]) ** 2.,
            axis=(0, 1, 2)) * _uct[np.concatenate([[i] * len(indexes[_i])
                                   for i, _i in enumerate(_uix)])])

    def trkij():
        return np.ma.sum(_uct * [
            np.ma.sum((data[..., indexes[_u].astype(int, casting='unsafe')]
            - _nanmean(data[..., indexes[_u].astype(
                int, casting='unsafe')])[..., None]) ** 2.)
            for _u in _uix])

    trkw_denom = len(np.concatenate(indexes)) - 1.
    trkij_denom = trkw_denom - indexes.shape[0] + 1.

    trka = trkw() / trkw_denom
    trkb = trkij() / trkij_denom
    trkx = trka - trkb

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        score = trkx / (trkx + trkb)
        if np.isnan(score):
            score = 1.
        return score


def _cv(indexes, data):
    mean_subject = _nanmean(data[..., indexes])
    zero_mean = np.isclose(mean_subject, 0.)
    local_copy = np.copy(data[..., indexes])
    mean_subject[zero_mean] = np.ma.masked
    local_copy[zero_mean, :] = np.ma.masked
    return np.ma.std(data[..., indexes], axis=-1) / mean_subject


def _corr(indexes, data):
    std_array = np.concatenate([
        np.nanstd(
            data[..., idxs.astype(int, casting='unsafe')], axis=-1)[..., None]
        for idxs in indexes], axis=-1)

    corr = np.prod([
        (data[..., idxs.astype(int, casting='unsafe')] -
         _nanmean(data[..., idxs.astype(int, casting='unsafe')])[..., None])
        for idxs in indexes], axis=0) / (
            float(len(indexes[0])) *
            np.prod(std_array, axis=-1)[..., None])

    corr[np.any(np.isclose(std_array, 0.), axis=-1)] = 1.

    return _nanmean(corr)

# Measure interfaces for significance computation

def _significance_cv(indexes, data, subjects, visits):
    return _nanmean(_cv(np.hstack(indexes), data), (0, 1, 2))


def _significance_corr(indexes, data, subjects, visits):
    return _nanmean(_corr(indexes, data), (0, 1, 2))

# Subjects and sessions identification

def _get_measures(args):
    measures = list(
        filter(lambda it: os.path.isdir(os.path.join(args.measures_dir, it)),
               os.listdir(args.measures_dir)))

    images = os.listdir(os.path.join(args.measures_dir, measures[0]))
    sub_ses = np.sort([i.split("___")[0] for i in images])
    subjects, ses_counts = np.unique([ss.split("_")[0] for ss in sub_ses],
                                     return_counts=True)

    return measures, subjects, ses_counts

# Image loading

def _load_images_4D(img_list):
    imgs = [nib.load(i) for i in img_list]
    return np.concatenate(
        tuple(i.get_fdata() if len(i.shape) == 4 else
              i.get_fdata()[..., None] for i in imgs),
        axis=-1), \
        imgs[0].affine


def _get_masks(args, n_subjects, n_sessions_per_subject):
    def _cast_to_mask(img):
        return img.get_fdata().astype(bool)

    if args.tissue_masks_dir is not None:
        masks = sorted([os.path.join(args.tissue_masks_dir, f)
                        for f in os.listdir(args.tissue_masks_dir)])
    else:
        masks = args.tissue_masks
        if masks is not None:
            masks = sorted(masks)


    if masks is None or len(masks) == 0:
        return None, 0, None
    elif len(masks) == 1:
        img = nib.load(masks[0])
        return _cast_to_mask(img), 1, img.shape[:3]
    elif len(masks) == n_subjects:
        shape = nib.load(masks[0]).shape[:3]
        return {
            os.path.basename(f).split("___")[0]: _cast_to_mask(nib.load(f))
            for f in masks
        }, 2, shape
    elif len(masks) == np.sum(n_sessions_per_subject):
        shape = nib.load(masks[0]).shape[:3]
        tmp = {
            os.path.basename(f).split("___")[0]: _cast_to_mask(nib.load(f))
            for f in masks
        }
        subjects = np.unique([k.split("_")[0] for k in tmp.keys()])
        tmp2 = {s: {} for s in subjects}
        for k in tmp.keys():
            sub, ses = k.split("_")
            tmp2[sub][ses] = tmp[k]
        return tmp2, 3, shape
    else:
        raise NotImplementedError("Must supply either one mask, a mask "
                                  "per subject or a mask per subject "
                                  "per session")


def _get_subject_mask(masks, subject, mask_type, none_shape):
    if mask_type == 0:
        return np.ones(none_shape, dtype=bool)
    if mask_type == 1:
        return np.copy(masks)
    elif mask_type == 2:
        return np.copy(masks[subject])
    else:
        return np.concatenate(
            [m[..., None] for m in masks[subject].values()], axis=-1)

# Missing data fixing

def _fix_missing_data(data, mask, strat):
    if len(mask.shape) == 3:
        if len(data.shape) > 3:
            mask = np.expand_dims(mask, -1)
    else:
        if strat == "exclude":
            data = np.ma.array(data, mask=~mask, fill_value=np.nan).filled()
        elif strat == "zero":
            data = np.ma.array(data, mask=~mask, fill_value=0.).filled()
        elif strat == "mean":
            data = np.ma.array(data, mask=~mask, fill_value=np.nan).filled()
            mean_series = _nanmean(data)
            for i, m in enumerate(np.moveaxis(mask, -1, 0)):
                data[~m, i] = mean_series[~m]
        elif not strat == "none":
            raise RuntimeError("Invalid strategy for fixing missing datapoints")

        mask = np.repeat(
            np.expand_dims(np.any(mask, axis=-1), -1), data.shape[-1], -1)

    return np.ma.array(data, mask=~mask)

# Significance helper functions

def _balance_visits(subjects, visits):
    def _fill(_d, v):
        if len(_d) < v:
            _dd = np.arange(0, v)
            _dd[_dd < np.min(_d)] = np.min(_d)
            _dd[_dd > np.max(_d)] = np.max(_d)
            for i in range(len(_d) - 1):
                _dd[(_d[i] < _dd) & (_dd < _d[i + 1])] = _d[i]
            return _dd
        return _d

    _s, _sct = np.unique(subjects, return_counts=True)
    _sct_max = np.max(_sct)
    ss, vs, _df = [], [], 0
    for _ss, _ssct in zip(_s, _sct):
        if _ssct <_sct_max:
            _sv = _fill(visits[subjects == _ss], _sct_max)
        else:
            _sv = visits[subjects == _ss]
        ss.extend([_ss] * _sct_max)
        vs.extend(_sv - 1 + _df)
        _df += _ssct

    return np.asarray(ss), np.array(vs, dtype=int)

# Significance threading fixture

_bst_th_data = None
def _init_bst_thread(_data, _subjects, _visits):
    global _bst_th_data
    _bst_th_data = (_data, _subjects, _visits)


def _run_bst_thread(indexes, _fn):
    global _bst_th_data
    return [_fn(indexes, *_bst_th_data)]

# Significance computation

def significance(
    fn, score, data, subjects, visits, ci,
    n=100,
    batch_size=cpu_count(),
    method="basic",
    paired=False,
    balanced=False,
    single_subject=False):

    balanced = balanced or paired
    subjects, visits = np.asarray(subjects), np.asarray(visits)
    if balanced:
        subjects, visits = _balance_visits(subjects, visits)

    with Pool(
        batch_size,
        initializer=_init_bst_thread,
        initargs=(data, subjects, visits)) as p:

        def _bootstrapper(*s_indexes, axis):
            coeff_on_sample = len(s_indexes[0].shape) > 1
            s_indexes = np.array(s_indexes).squeeze()
            if balanced:
                if len(s_indexes.shape) == 2:
                    s_indexes = s_indexes[:, None, :]
                s_indexes = np.moveaxis(s_indexes, 0, 1)
            else:
                if len(s_indexes.shape) == 1:
                    s_indexes = s_indexes[None, :]

            _fn = partial(_run_bst_thread, _fn=fn)
            start = time.time()
            res = p.starmap(_fn, ((
                np.array([np.array(i, dtype=int) for i in idxs if i.size > 0],
                dtype=object),) for idxs in s_indexes))

            if coeff_on_sample:
                res = np.concatenate(res)[None, ...]
            else:
                res = np.concatenate(res)
            print("Pool processing time : {} s".format(time.time() - start))
            return res

        data_range = np.arange(0, len(subjects))
        boot_data = [data_range[subjects == s] for s in np.unique(subjects)]
        boot_data = boot_data if paired or single_subject else [boot_data]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",
                                  category=np.VisibleDeprecationWarning)
            res = bootstrap(
                boot_data, _bootstrapper,
                method=method,
                paired=paired,
                confidence_level=ci,
                n_resamples=n)

            p.close()

        lb, ub = res.confidence_interval
        lb = lb.item() if isinstance(lb, np.ndarray) else lb
        ub = ub.item() if isinstance(ub, np.ndarray) else ub
        z = score / res.standard_error
        p = 1. - erf(z / np.sqrt(2.))
        return lb, ub, res.standard_error.item(), p.item()

def _significance_for_average(subjects, p_ceil):
    def _no_na_reduce(_vals, _idx, _red_fn, _na_val="NA", _all_na_val="NA"):
        _no_na = [_v[_idx] for _v in _vals if not _v[_idx] == _na_val]
        if len(_no_na) == 0:
            return _all_na_val
        return _red_fn(_no_na)

    lb = _no_na_reduce(subjects.values(), 1, min)
    ub = _no_na_reduce(subjects.values(), 2, max)
    se = _no_na_reduce(subjects.values(), 3, np.mean)
    p = _no_na_reduce(subjects.values(), 4, lambda a: a)
    if not p == "NA":
        p = _no_na_reduce(
            [[pp] for pp in p], 0, max,
            "< {:.2f}".format(p_ceil),
            "< {:.2f}".format(p_ceil))

    return lb, ub, se, p

# Parsing inputs per measure and compute

def compute_variation(args):
    p_ceil = 1. - args.conf_interval
    measures, subjects, ses_counts = _get_measures(args)
    intra_subject = {}
    inter_subject = {}

    masks, mask_type, shape_3d = _get_masks(args, len(subjects), ses_counts)

    for m in measures:
        all_mask = None
        images = sorted(os.listdir(os.path.join(args.measures_dir, m)))
        avg_subjects, affine = [], None
        intra_m = {}
        for subject in subjects:
            s_pattern = re.compile(
                r"{}_ses[-]?[0-9]+.+\.nii\.gz".format(subject))
            s_images, affine = _load_images_4D([
                os.path.join(args.measures_dir, m, f)
                for f in filter(lambda ff: s_pattern.match(ff), images)])

            mask = _get_subject_mask(masks, subject, mask_type, shape_3d)
            s_images = _fix_missing_data(s_images, mask, args.missing_data)

            if s_images.shape[-1] > 1:
                intra_cv = _cv(np.arange(0, s_images.shape[-1]), s_images)
                nib.save(nib.Nifti1Image(intra_cv, affine),
                         os.path.join(
                            args.output_dir,
                            "{}_{}_intra_cv.nii.gz".format(
                                subject, m)))
                intra_cv = _nanmean(intra_cv, (0, 1, 2))
                lb, ub, se, p = "NA", "NA", "NA", "NA"
                intra_m[subject] = (intra_cv, lb, ub, se, p)
                avg_subjects.append(_nanmean(s_images))
            else:
                avg_subjects.append(s_images.squeeze())

            if all_mask is None:
                all_mask = np.any(~s_images.mask, axis=-1)[..., None]
            elif mask_type not in [0, 1]:
                all_mask = np.concatenate(
                    (all_mask, np.any(~s_images.mask, axis=-1)[..., None]),
                    axis=-1)

        intra_subject[m] = intra_m
        avg_subjects = np.concatenate(
            [img[..., None] for img in avg_subjects], axis=-1)

        avg_subjects = _fix_missing_data(
            avg_subjects, all_mask.squeeze(), args.missing_data)

        inter_cv = _cv(np.arange(0, avg_subjects.shape[-1]), avg_subjects)
        nib.save(nib.Nifti1Image(inter_cv, affine),
                 os.path.join(
                    args.output_dir,
                    "{}_inter_cv.nii.gz".format(m)))

        inter_cv = _nanmean(inter_cv, (0, 1, 2))
        lb, ub, se, p = "NA", "NA", "NA", "NA"
        inter_subject[m] = (inter_cv, lb, ub, se, p)

    with open(os.path.join(args.output_dir, "inter-subject.csv"), "w+") as f:
        f.write("measure,cv,lb,ub,se,p\n")
        for m, (cv, lb, ub, se, p) in inter_subject.items():
            f.write("{},{},{},{},{},{}\n".format(m, cv, lb, ub, se, p))

    if len(intra_subject) > 0:
        with open(os.path.join(args.output_dir, "intra-subject.csv"),
                  "w+") as f:
            f.write("subject,measure,cv,lb,ub,se,p\n")
            for m, subjects in intra_subject.items():
                for subject, (cv, lb, ub, se, p) in subjects.items():
                    f.write("{},{},{},{},{},{},{}\n".format(
                        subject, m, cv, lb, ub, se, p))

                avg = _nanmean(list([v[0] for v in subjects.values()]))
                lb, ub, se, p = "NA", "NA", "NA", "NA"
                if args.significance:
                    lb, ub, se, p = _significance_for_average(subjects, p_ceil)

                f.write("average,{},{},{},{},{},{}\n".format(
                    m, avg, lb, ub, se, p))


def compute_i2c2(args):
    measures, subjects, ses_counts = _get_measures(args)
    i2c2 = {}

    masks, mask_type, shape_3d = _get_masks(args, len(subjects), ses_counts)
    p_ceil = 1. - args.conf_interval

    for m in measures:
        images = sorted(os.listdir(os.path.join(args.measures_dir, m)))

        masks_sub = (_get_subject_mask(masks, s, mask_type, shape_3d)
                     for s in subjects)
        masks_sub = (m if len(m.shape) == 4 else m[..., None]
                     for m in masks_sub)
        masks_sub = (np.repeat(m, s, -1) if m.shape[-1] == 1 else m
                     for m, s in zip(masks_sub, ses_counts))

        data, subs, visits = None, [], []
        for subject, sc in zip(subjects, ses_counts):
            s_pattern = re.compile(
                r"{}_ses[-]?[0-9]+.+\.nii\.gz".format(subject))
            s_images, affine = _load_images_4D([
                os.path.join(args.measures_dir, m, f)
                for f in filter(lambda ff: s_pattern.match(ff), images)])

            subs.extend([subject] * sc)
            visits.extend([i + 1 for i in range(sc)])
            mask = _get_subject_mask(masks, subject, mask_type, shape_3d)
            masked_images = _fix_missing_data(s_images, mask, args.missing_data)

            if data is None:
                data = masked_images
            else:
                data = np.ma.concatenate((data, masked_images), axis=-1)

        subs = np.array(subs)
        _, n_visits = np.unique(subs, return_counts=True)
        score = _i2c2(
            np.array([np.arange(i, i + j) for i, j in zip(
                np.cumsum([0] + n_visits[:-1].tolist()), n_visits
            )], dtype=object),
            data,
            np.asarray(subs),
            np.arange(0, data.shape[-1]))

        lb, ub, se, p = "NA", "NA", "NA", "NA"
        if args.significance:
            lb, ub, se, p = significance(
                _i2c2, score, data, subs, visits,
                args.conf_interval,
                args.resample,
                args.nb_threads,
                args.method)

            if not isinstance(p, str) and p < p_ceil:
                p = pp = "< {:.2f}".format(p_ceil)
            else:
                pp = "{:.4f}".format(p)

            print("Measure : {} | I2C2 score : {:.4f} | CI : {:.4f} - {:.4f} "
                  "| SE : {:.3E} | p-value : {}".format(
                      m, score, lb, ub, se, pp))

        i2c2[m] = (score, lb, ub, se, p)

    with open(os.path.join(args.output_dir, "i2c2.csv"), "w+") as f:
        f.write("measure,i2c2,cl,cu,se,p < {}\n".format(p_ceil))
        for m, (coeff, lb, ub, se, p) in i2c2.items():
            f.write("{},{},{},{},{},{}\n".format(
                m, coeff, lb, ub, se, p
            ))


def compute_correlation(args):
    p_ceil = 1. - args.conf_interval
    measures, subjects, ses_counts = _get_measures(args)
    average_correlations = {}

    masks, mask_type, shape_3d = _get_masks(args, len(subjects), ses_counts)

    for m1, m2 in itertools.combinations(measures, 2):
        images_m1 = sorted(os.listdir(os.path.join(args.measures_dir, m1)))
        images_m2 = sorted(os.listdir(os.path.join(args.measures_dir, m2)))
        m1_m2_avg_corr = {}
        for subject in subjects:
            s_pattern = re.compile(
                r"{}_ses[-]?[0-9]+.+\.nii\.gz".format(subject))
            s_images_m1, affine = _load_images_4D([
                os.path.join(args.measures_dir, m1, f)
                for f in filter(lambda ff: s_pattern.match(ff), images_m1)])
            s_images_m2, _ = _load_images_4D([
                os.path.join(args.measures_dir, m2, f)
                for f in filter(lambda ff: s_pattern.match(ff), images_m2)])

            mask = _get_subject_mask(masks, subject, mask_type, shape_3d)
            s_images_m1 = _fix_missing_data(
                s_images_m1, mask, args.missing_data)
            s_images_m2 = _fix_missing_data(
                s_images_m2, mask, args.missing_data)

            data = np.concatenate((s_images_m1, s_images_m2), axis=-1)
            indexes = (np.arange(0, s_images_m1.shape[-1]),
                       s_images_m1.shape[-1] + np.arange(
                           0, s_images_m2.shape[-1]))

            corr = _corr(indexes, data)

            nib.save(nib.Nifti1Image(corr, affine),
                     os.path.join(args.output_dir,
                                  "{}_corr_{}_{}.nii.gz".format(
                                      subject, m1, m2)))

            score = _nanmean(corr, (0, 1, 2))
            lb, ub, se, p = "NA", "NA", "NA", "NA"
            if args.significance and s_images_m1.shape[-1] > 1:
                lb, ub, se, p = significance(
                    _significance_corr,
                    score,
                    data,
                    np.concatenate([np.repeat(m1, len(indexes[0])),
                                    np.repeat(m2, len(indexes[1]))]),
                    np.concatenate(
                        [np.arange(1, len(indexes[0]) + 1, dtype=int),
                         np.arange(1, len(indexes[1]) + 1, dtype=int)]),
                    args.conf_interval,
                    args.resample,
                    args.nb_threads,
                    args.method,
                    paired=True)

                if not isinstance(p, str) and p < p_ceil:
                    p = pp = "< {:.2f}".format(p_ceil)
                else:
                    pp = "{:.4f}".format(p)

                print("Measures : {} - {} | Subject : {} | "
                      "Correlation score : {:.4f} | CI : {:.4f} - {:.4f} "
                      "| SE : {:.3E} | p-value : {}".format(
                          m1, m2, subject, score, lb, ub, se, pp))

            m1_m2_avg_corr[subject] = (score, lb, ub, se, p)

        average_correlations["{}-{}".format(m1, m2)] = m1_m2_avg_corr

    with open(os.path.join(args.output_dir, "avg-correlation.csv"), "w+") as f:
        f.write("subject,measure 1,measure 2,avg pearson,lb,ub,se,p\n")
        for m, subjects in average_correlations.items():
            for subject, (pearson, lb, ub, se, p) in subjects.items():
                f.write(
                    "{},{},{},{},{},{},{},{}\n".format(
                        subject, *m.split("-"), pearson, lb, ub, se, p))

            avg = _nanmean(list(v[0] for v in subjects.values()))
            lb, ub, se, p = "NA", "NA", "NA", "NA"
            if args.significance:
                lb, ub, se, p = _significance_for_average(subjects, p_ceil)

            f.write("average,{},{},{},{},{},{},{}\n".format(
                *m.split("-"), avg, lb, ub, se, p
            ))

if __name__ == "__main__":
    p = _create_parser()
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.disable_corr_coeff:
        compute_correlation(args)

    if not args.disable_var_coeff:
        compute_variation(args)

    if not args.disable_i2c2_coeff:
        compute_i2c2(args)
