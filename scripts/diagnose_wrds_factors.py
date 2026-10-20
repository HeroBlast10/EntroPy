"""Produce comparable WRDS factor scorecards without selecting on full history."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from quant_platform.core.data.returns import forward_total_returns
from quant_platform.core.evaluation.factor_diagnostics import factor_diagnostics
from quant_platform.core.utils.provenance import fingerprint,write_run_metadata,WRDS_CODE_PATHS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT/"data/wrds/batch2")
    parser.add_argument("--missing-terminal-return",choices=["total_loss","zero_return"],required=True)
    args=parser.parse_args()
    dest=args.root/"diagnostics"/args.missing_terminal_return; dest.mkdir(parents=True,exist_ok=True)
    f=pd.read_parquet(args.root/"factors.parquet")
    prices_path=ROOT/"data/wrds/p0/prices.parquet"
    p=pd.read_parquet(prices_path,columns=["date","ticker","close","total_return","is_terminal"])
    for horizon in (1,5,20,60):
        labels=forward_total_returns(p,horizon,missing_terminal_return=args.missing_terminal_return)
        f=f.merge(labels[["date","ticker","forward_return"]].rename(columns={"forward_return":f"fwd_{horizon}"}),
                  on=["date","ticker"],how="left",validate="one_to_one")
    eligible=f[f.risk_eligible & f.date.ge("2010-01-01")].copy()
    results=factor_diagnostics(eligible,pd.read_csv(args.root/"factor_catalog.csv"))
    for name,frame in results.items():
        if name=="daily_ic": frame.to_parquet(dest/f"{name}.parquet")
        else: frame.to_csv(dest/f"{name}.csv",index=name=="rank_correlation")
    write_run_metadata(dest/"run_metadata.json",stage="wrds_factor_diagnostics",code_paths=WRDS_CODE_PATHS,
                       inputs={"factors":args.root/"factors.parquet","prices":prices_path,"catalog":args.root/"factor_catalog.csv"},
                       params={"horizons":[1,5,20,60],"terminal_policy":args.missing_terminal_return,
                               "directions":"predeclared FactorMeta; never flipped by realized IC",
                               "selection_use":"descriptive only; OOF schema uses first training window"},
                       extra={"eligible_rows":len(eligible),"factor_count":len(results["quality"]),
                              "summary_rows":len(results["summary"]),"fundamentals_is_point_in_time":False})


if __name__=="__main__": main()
