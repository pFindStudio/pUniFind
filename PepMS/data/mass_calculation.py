import pickle

import numpy as np
import torch

PTM_NL_priority = {}
PTM_NL_priority["Phospho[S]"] = 1e8

PTM_NL_priority["Phospho[T]"] = 1e7

PTM_NL_priority["GG[K]"] = 1e6

PTM_NL_priority["Oxidation[M]"] = 1e5

PPM = 1 / 1000000
ASCII = 128


class BaseMass:
    def __init__(self):
        self.mass_H = 1.0078250321
        self.mass_O = 15.9949146221
        self.mass_N = 14.0030740052
        self.mass_C = 12.00
        self.mass_isotope = 1.003
        self.mass_proton = 1.007276

        self.mass_H2O = self.mass_H * 2 + self.mass_O
        self.mass_CO = self.mass_C + self.mass_O
        self.mass_CO2 = self.mass_C + self.mass_O * 2
        self.mass_NH = self.mass_N + self.mass_H
        self.mass_NH3 = self.mass_N + self.mass_H * 3
        self.mass_HO = self.mass_H + self.mass_O


def pearson_correlation(x: torch.Tensor, y: torch.Tensor):
    """Compute pearson correlation between 2 batches of 1-D tensors

    Parameters
    ----------
    x : torch.Tensor
        Shape (Batch, n)

    y : torch.Tensor
        Shape (Batch, n)

    """
    return torch.cosine_similarity(x - x.mean(), y - y.mean(), dim=0)


def spectral_angle(cos):
    cos[cos > 1] = 1
    return 1 - 2 * torch.arccos(cos) / np.pi


def _get_ranks(x: torch.Tensor, device) -> torch.Tensor:
    sorted_idx = x.argsort(dim=1)
    flat_idx = (
        sorted_idx + torch.arange(x.size(0), device=device).unsqueeze(1) * x.size(1)
    ).flatten()
    ranks = torch.zeros_like(flat_idx)
    ranks[flat_idx] = (
        torch.arange(x.size(1), device=device)
        .unsqueeze(0)
        .repeat(x.size(0), 1)
        .flatten()
    )
    ranks = ranks.reshape(x.size())
    ranks[x == 0] = 0
    return ranks


def spearman_correlation(x: torch.Tensor, y: torch.Tensor, device):
    """Compute spearman correlation between 2 batches of 1-D tensors

    Parameters
    ----------
    x : torch.Tensor
        Shape (Batch, n)

    y : torch.Tensor
        Shape (Batch, n)

    """
    x_rank = _get_ranks(x, device).to(torch.float32)
    y_rank = _get_ranks(y, device).to(torch.float32)

    n = x.size(1)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2), dim=1)
    down = n * (n**2 - 1.0)
    return 1.0 - (upper / down)


class PeptideIonCalculator:

    def __init__(
        self, modification_meta_dict_path=r"./consts/modification_meta_dict.pkl"
    ):
        self.base_mass = BaseMass()

        self.AAMass = np.zeros(ASCII, dtype=np.float64)
        self.AAMass[ord("A")] = 71.037114
        self.AAMass[ord("C")] = 103.009185
        self.AAMass[ord("D")] = 115.026943
        self.AAMass[ord("E")] = 129.042593
        self.AAMass[ord("F")] = 147.068414
        self.AAMass[ord("G")] = 57.021464
        self.AAMass[ord("H")] = 137.058912
        self.AAMass[ord("I")] = 113.084064
        self.AAMass[ord("J")] = 114.042927
        self.AAMass[ord("K")] = 128.094963
        self.AAMass[ord("L")] = 113.084064
        self.AAMass[ord("M")] = 131.040485
        self.AAMass[ord("N")] = 114.042927
        self.AAMass[ord("P")] = 97.052764
        self.AAMass[ord("Q")] = 128.058578
        self.AAMass[ord("R")] = 156.101111
        self.AAMass[ord("S")] = 87.032028
        self.AAMass[ord("T")] = 101.047679
        self.AAMass[ord("V")] = 99.068414
        self.AAMass[ord("W")] = 186.079313
        self.AAMass[ord("Y")] = 163.06332
        self.modification_meta_dict = pickle.load(
            open(modification_meta_dict_path, "rb")
        )

        self.modification_meta_dict["Ammonia-loss"] = self.modification_meta_dict[
            "Ammonia-loss[AnyN-termC]"
        ]

        # self.ModMass = {}
        # for modname, modinfo in mod_dict.items():
        #     modinfo = modinfo.split(' ')
        #     modmass = float(modinfo[2])
        #     mod_neutral_loss = 0
        #     if modinfo[4] != '0':
        #         mod_neutral_loss = float(modinfo[5])
        #     self.ModMass[modname] = (modmass, mod_neutral_loss)

    def get_aamass(self, aa):
        return self.AAMass[ord(aa)]

    def set_aamass(self, aa, mass):
        self.AAMass[ord(aa)] = mass

    def set_aa_label(self, aa, label_mass):
        self.AAMass[ord(aa)] += label_mass

    def calc_aamass_cumsum(self, peptide):
        return np.cumsum(self.AAMass[[ord(aa) for aa in peptide]])

    def calc_modification_mass(self, peptide, modinfo):
        modmass = np.zeros(len(peptide) + 2)
        if modinfo == "":
            return modmass, None, None

        lossmass = np.zeros(len(peptide) + 2)
        retmods = [""] * (len(peptide) + 2)

        # items = modinfo.strip(";").split(";")

        # for mod in items:
        #     strSite, modname = mod.split(",")
        #     site = int(strSite)
        #     modlist.append((site, modname))
        modlist = modinfo
        # try:
        for site, modname in modlist:
            _mono, _loss = (
                self.modification_meta_dict[modname][1],
                self.modification_meta_dict[modname][3],
            )
            # assert 0, (self.modification_meta_dict[modname], _mono, _loss)
            modmass[site] = _mono
            lossmass[site] = _loss
            if modname in PTM_NL_priority:
                retmods[site] = modname
            else:
                retmods[site] = ""  # 0 PTM_NL_priority
        # except:
        #     breakpoint()
        #     for site, modname, _ in modlist:
        #         _mono, _loss = self.modification_meta_dict[modname][1], self.modification_meta_dict[modname][3]
        #         # assert 0, (self.modification_meta_dict[modname], _mono, _loss)
        #         modmass[site] = _mono
        #         lossmass[site] = _loss
        #         if modname in PTM_NL_priority:
        #             retmods[site] = modname
        #         else:
        #             retmods[site] = ""  # 0 PTM_NL_priority
        return np.cumsum(modmass), lossmass, retmods

    # def calc_pepmass(self, peptide, modinfo):
    #     if not modinfo:
    #         return sum(self.AAMass[[ord(aa) for aa in peptide]]) + self.base_mass.mass_H2O
    #     else:
    #         modmass = sum([self.ModMass[onemod[onemod.find(',')+1:]][0] for onemod in modinfo.strip(";").split(";")])
    #         return sum(self.AAMass[[ord(aa) for aa in peptide]]) + modmass + self.base_mass.mass_H2O

    def calc_b_ions_and_pepmass(self, peptide, mod_cumsum):
        aa_cumsum = self.calc_aamass_cumsum(peptide)
        b_ions = aa_cumsum[:-1] + mod_cumsum[1:-2]
        pepmass = aa_cumsum[-1] + mod_cumsum[-1] + self.base_mass.mass_H2O
        return b_ions, pepmass

    def calc_y_from_b(self, bions, pepmass):
        return pepmass - bions

    def calc_by_and_pepmass(self, peptide, modinfo, max_charge=2):
        mod_cumsum, _, _ = self.calc_modification_mass(peptide, modinfo)
        bions, pepmass = self.calc_b_ions_and_pepmass(peptide, mod_cumsum)
        yions = self.calc_y_from_b(bions, pepmass)

        ions = [
            bions / charge + self.base_mass.mass_proton
            for charge in range(1, max_charge + 1)
        ]
        ions.extend(
            [
                yions / charge + self.base_mass.mass_proton
                for charge in range(1, max_charge + 1)
            ]
        )

        return np.array(ions).T, pepmass  # .T, from 4xn to nx4, time step first

    def calc_pepmass_and_ions_from_iontypes(
        self, peptide, modinfo, ion_types, max_charge, shift=True
    ):
        mod_cumsum, modloss_list, modname_list = self.calc_modification_mass(
            peptide, modinfo
        )
        bions, pepmass = self.calc_b_ions_and_pepmass(peptide, mod_cumsum)
        yions = self.calc_y_from_b(bions, pepmass)
        bions_ = np.append(bions, 0.0)
        # print(bions, yions)
        if shift:
            yions_ = np.insert(yions, 0, 0.0)
        else:
            yions_ = np.append(yions, 0.0)
        # print(bions, yions)
        ions = []
        for _type in ion_types:
            _type = _type.format("")
            if _type == "b":
                ions.extend(
                    [
                        bions_ / charge + self.base_mass.mass_proton
                        for charge in range(1, max_charge + 1)
                    ]
                )
            elif _type == "y":
                ions.extend(
                    [
                        yions_ / charge + self.base_mass.mass_proton
                        for charge in range(1, max_charge + 1)
                    ]
                )
            elif _type == "b-NH3":
                nh3_ions = self.calc_NH3_loss(bions_)
                ions.extend(
                    [
                        nh3_ions / charge + self.base_mass.mass_proton
                        for charge in range(1, max_charge + 1)
                    ]
                )
            elif _type == "y-NH3":
                nh3_ions = self.calc_NH3_loss(yions_)
                ions.extend(
                    [
                        nh3_ions / charge + self.base_mass.mass_proton
                        for charge in range(1, max_charge + 1)
                    ]
                )
            elif _type == "b-H2O":
                h2o_ions = self.calc_H2O_loss(bions_)
                ions.extend(
                    [
                        h2o_ions / charge + self.base_mass.mass_proton
                        for charge in range(1, max_charge + 1)
                    ]
                )
            elif _type == "y-H2O":
                h2o_ions = self.calc_H2O_loss(yions_)
                ions.extend(
                    [
                        h2o_ions / charge + self.base_mass.mass_proton
                        for charge in range(1, max_charge + 1)
                    ]
                )
            elif _type.lower() == "b-modloss":
                bloss = self.calc_Nterm_modloss(bions, modloss_list, modname_list)
                bloss = np.append(bloss, 0.0)
                ions.extend(
                    [
                        bloss / charge + self.base_mass.mass_proton
                        for charge in range(1, max_charge + 1)
                    ]
                )
            elif _type.lower() == "y-modloss":
                yloss = self.calc_Cterm_modloss(yions, modloss_list, modname_list)
                if shift:
                    yloss = np.insert(yloss, 0, 0.0)
                else:
                    yloss = np.append(yloss, 0.0)
                ions.extend(
                    [
                        yloss / charge + self.base_mass.mass_proton
                        for charge in range(1, max_charge + 1)
                    ]
                )
        return pepmass, np.array(ions).T  # shape should be (seq_len, ion_type)

    def sum_intensity_in_range(self, A, B, interval):
        min_val, max_val = interval
        # 找到落在区间内的索引
        indices = torch.where((A >= min_val) & (A <= max_val))
        ret_flag = (A >= min_val) & (A <= max_val)
        # 提取落在区间内的强度值，并求和
        # print(B[indices])
        if len(B[indices]) == 0:
            return 0, ret_flag
        # print(B[indices], len(B[indices]))
        sum_intensity = torch.sum(B[indices])
        return sum_intensity, ret_flag

    def get_spectrum_prediction_label(
        self,
        peptide,
        mz_array,
        intensity,
        mods,
        ion_types,
        simple=False,
        max_charges=4,
        PPM_THRESHOLD=20,
        flag_ITMS=False,
        shift=True,
        return_mz=False,
    ):
        peptide_mass, ions = self.calc_pepmass_and_ions_from_iontypes(
            peptide, mods, ion_types, max_charges, shift=shift
        )
        mask = ions < 2
        ions[mask] = 0
        error_tol = PPM_THRESHOLD * PPM

        intens = np.zeros(ions.shape)
        # print(len(peptide), ions)
        ion_peak_flag = torch.zeros(
            mz_array.shape[0],
        )
        ion_peak_class = torch.zeros(
            mz_array.shape[0],
        )
        ion_res_num = torch.zeros(
            mz_array.shape[0],
        )
        res_idx = torch.zeros(
            mz_array.shape[0],
        )
        mz_array = torch.Tensor(mz_array)
        intensity = torch.Tensor(intensity)
        # print(ions.shape, len(ion_types))
        for i in range(ions.shape[0]):
            for j in range(ions.shape[1]):
                if ions[i, j] <= 0:
                    pass
                else:
                    inten, indices = self.sum_intensity_in_range(
                        mz_array,
                        intensity,
                        (
                            ions[i, j] - error_tol * ions[i, j],
                            ions[i, j] + error_tol * ions[i, j],
                        ),
                    )
                    # print(indices)
                    intens[i, j] = inten
                    if not simple:
                        # print(ion_peak_flag.shape, indices.reshape(ion_peak_flag.shape).shape)
                        ion_peak_flag = ion_peak_flag + indices
                        ion_peak_class = (
                            ion_peak_class
                            + (indices * (j + 1)) * ~ion_peak_class.bool()
                        )
                        assert torch.all(ion_peak_class < 33)
                        if j < (len(ion_types) * max_charges) / 2:
                            ion_res_num += indices * (i + 1) * ~ion_res_num.bool()
                            # if torch.any(indices > 0):
                            #     print(">?????", i, j)
                            # print(">>>>", ion_res_num)
                            if torch.any(ion_res_num >= 100):
                                print(">>", ion_res_num)
                            res_idx += indices * i * ~res_idx.bool()
                        else:
                            if shift:
                                ion_res_num += (
                                    indices * (ions.shape[0] - i) * ~ion_res_num.bool()
                                )
                                if torch.any(ion_res_num >= 100):
                                    print("<<", ion_res_num)
                                res_idx += indices * i * ~res_idx.bool()
                            elif i > 0:
                                ion_res_num += (
                                    indices
                                    * (ions.shape[0] - i - 1)
                                    * ~ion_res_num.bool()
                                )
                                if torch.any(ion_res_num >= 100):
                                    print("<<", ion_res_num)
                                res_idx += indices * (i + 1) * ~res_idx.bool()

                        assert torch.all(ion_res_num < 110), (peptide, ion_res_num)
        if not return_mz:
            return (
                intens,
                ion_peak_flag > 0,
                ion_peak_class,
                ion_res_num,
                res_idx,
                error_tol,
            )
        else:
            return (
                intens,
                ion_peak_flag > 0,
                ion_peak_class,
                ion_res_num,
                res_idx,
                error_tol,
                ions,
            )

    def cal_features(
        self,
        pred: torch.Tensor,
        intensity: torch.Tensor,
        threadshold=1e-4,
    ):
        pred = pred[: intensity.shape[0]]
        pred_mask = (pred > threadshold).reshape(-1)
        # print(intensity.shape, pred.shape, pred_mask.shape)
        pred = pred.reshape(-1)[pred_mask]
        intensity = intensity.reshape(-1)[pred_mask]
        pred = torch.Tensor(pred)
        intensity = torch.Tensor(intensity)
        # max_val =
        # intensity = intensity / torch.max(intensity, dim=0)
        pcc = pearson_correlation(pred, intensity)
        # assert 0, (pred, intensity)
        # print(pred.shape, intensity.shape)
        cos = torch.cosine_similarity(pred, intensity, dim=0)
        sa = spectral_angle(cos)
        spc = spearman_correlation(
            pred.unsqueeze(0), intensity.unsqueeze(0), pred.device
        ).squeeze(0)
        # return {
        #     "pcc": pcc,
        #     "cos": cos,
        #     "sa": sa,
        #     "spc": spc,
        # }
        return torch.Tensor([pcc, cos, sa, spc])

    def cal_inference_feature(
        self,
        mz,
        intensity,
        spectrum_preds,
        peptides,
        modinfos,
        ion_types=[
            "b",
            "b-NH3",
            "b-H2O",
            "b-ModLoss",
            "y",
            "y-NH3",
            "y-H2O",
            "y-ModLoss",
        ],
        max_charge=4,
        shift=True,
    ):
        features = []
        for i in range(len(spectrum_preds)):
            intens, _, _, _, _, _, ions = self.get_spectrum_prediction_label(
                peptides[i],
                mz,
                intensity,
                modinfos[i],
                ion_types,
                simple=True,
                return_mz=True,
            )
            features.append(self.cal_features(spectrum_preds[i], intens))
            if i == 0:
                pred_mz = ions
                pred_intensity = spectrum_preds[0, : len(peptides[0])]
                assert (
                    pred_mz.shape[0] == pred_intensity.shape[0] == len(peptides[0])
                ), (pred_mz.shape, pred_intensity.shape, len(peptides[0]))

        return features, pred_mz.reshape(-1), pred_intensity.reshape(-1)

    def calc_a_from_b(self, bions):
        return bions - self.base_mass.mass_CO

    def calc_c_from_b(self, bions):
        return bions + self.base_mass.mass_NH3

    def calc_z_from_b(self, bions, pepmass):
        return (pepmass - self.base_mass.mass_NH3 + self.base_mass.mass_H) - bions

    def calc_H2O_loss(self, ions):
        return ions - self.base_mass.mass_H2O

    def calc_NH3_loss(self, ions):
        return ions - self.base_mass.mass_NH3

    def calc_Nterm_modloss(self, ions, modloss_list, modname_list):
        ret = np.zeros(len(ions), dtype=float)
        if modloss_list is None or np.all(modloss_list == 0):
            return ret

        loss_nterm = modloss_list[0]
        prev_prior = (
            PTM_NL_priority[modname_list[0]]
            if modname_list[0] in PTM_NL_priority
            else 0
        )

        for i in range(len(ions)):
            if modloss_list[i + 1] != 0:
                if modname_list[i + 1] in PTM_NL_priority:
                    if PTM_NL_priority[modname_list[i + 1]] >= prev_prior:
                        loss_nterm = modloss_list[i + 1]
                        prev_prior = PTM_NL_priority[modname_list[i + 1]]
                elif prev_prior == 0:
                    loss_nterm = modloss_list[i + 1]

            if loss_nterm != 0:
                ret[i] = ions[i] - loss_nterm
            else:
                ret[i] = 0.0
        return ret

    def calc_Cterm_modloss(self, ions, modloss_list, modname_list):
        ret = np.zeros(len(ions), dtype=float)
        if modloss_list is None or np.all(modloss_list == 0):
            return ret

        last_idx = len(modloss_list) - 1
        loss_cterm = modloss_list[last_idx]
        prev_prior = (
            PTM_NL_priority[modname_list[last_idx]]
            if modname_list[last_idx] in PTM_NL_priority
            else 0
        )

        for i in range(len(ions) - 1, -1, -1):
            if modloss_list[i + 2] != 0:
                if modname_list[i + 2] in PTM_NL_priority:
                    if PTM_NL_priority[modname_list[i + 2]] >= prev_prior:
                        loss_cterm = modloss_list[i + 2]
                        prev_prior = PTM_NL_priority[modname_list[i + 2]]
                elif prev_prior == 0:
                    loss_cterm = modloss_list[i + 2]

            if loss_cterm != 0:
                ret[i] = ions[i] - loss_cterm
            else:
                ret[i] = 0.0
        return ret


if __name__ == "__main__":
    peptide = "AAAAACCCC"
    ion_types = ["b", "b-NH3", "b-H2O", "b-ModLoss", "y", "y-NH3", "y-H2O", "y-ModLoss"]
    modinfo = []
    max_charge = 2

    cla = PeptideIonCalculator()

    pepmass, ions = cla.calc_pepmass_and_ions_from_iontypes(
        peptide, modinfo, ion_types, max_charge
    )
    print(pepmass, ions)
