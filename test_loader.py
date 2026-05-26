from data_loader import get_realtime_df
from data_loader import compute_regularite_metrics

df = get_realtime_df()
metrics = compute_regularite_metrics(df)

print(metrics)                                      