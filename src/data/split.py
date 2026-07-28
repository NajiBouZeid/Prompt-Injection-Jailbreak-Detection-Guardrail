from sklearn.model_selection import train_test_split
import pandas as pd


def stratified_split(
    df: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/val/test, stratified on label + source_dataset jointly.

    Stratifying on label alone would keep the overall 63/37 class balance in
    every split, but could still let one split end up with a skewed mix of
    sources (e.g. proportionally too little qualifire/neuralchemy, since they
    are much smaller than necent). Stratifying on the combined key preserves
    both the label balance and the source mix in every split.
    """
    strata = df["source_dataset"] + "_" + df["label"].astype(str)

    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        stratify=strata,
        random_state=random_state,
    )

    temp_strata = strata.loc[temp_df.index]
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (val_size + test_size),
        stratify=temp_strata,
        random_state=random_state,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
