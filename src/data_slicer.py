import os
import json
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta


class DataSlice:
    """Container for one time slice."""

    def __init__(self, start_time, end_time, df_slice, user_features=None, topic_features=None):
        self.start_time = start_time
        self.end_time = end_time
        self.df_slice = df_slice
        self.user_features = user_features
        self.topic_features = topic_features


class DataSlicer:
    """Time-based data slicer."""

    def __init__(self, input_dir, output_dir, interval_days=7, min_slice_size=50):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.interval_days = interval_days
        self.min_slice_size = min_slice_size

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_data(self, filename):
        """Load dataset (CSV or PKL)."""
        filepath = os.path.join(self.input_dir, filename)
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                df = pickle.load(f)
        else:
            raise ValueError("Unsupported file format.")
        return df

    def parse_time_column(self, df, time_col):
        """Parse timestamps to datetime, sort, and drop NaT."""
        if not np.issubdtype(df[time_col].dtype, np.datetime64):
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=[time_col])
        df = df.sort_values(by=time_col)
        return df

    def create_slices(self, df, time_col):
        """Create fixed-length time slices."""
        slices = []
        start_time = df[time_col].min()
        end_time = df[time_col].max()
        current_start = start_time

        while current_start < end_time:
            current_end = current_start + timedelta(days=self.interval_days)
            df_slice = df[(df[time_col] >= current_start) & (df[time_col] < current_end)]

            # Skip small slices
            if len(df_slice) >= self.min_slice_size:
                slices.append(DataSlice(current_start, current_end, df_slice))

            current_start = current_end

        return slices

    def _save_single_slice(self, slice_obj, slice_index, dataset_name):
        """Persist one slice under data/processed/<dataset>/slicer/<idx>."""
        # Build target directory
        base_processed_dir = os.path.join('data', 'processed', dataset_name, 'slicer')
        os.makedirs(base_processed_dir, exist_ok=True)
        slice_dir = os.path.join(base_processed_dir, str(slice_index))
        os.makedirs(slice_dir, exist_ok=True)

        # Save data.csv
        data_csv_path = os.path.join(slice_dir, 'data.csv')
        slice_obj.df_slice.to_csv(data_csv_path, index=False)

        # Save meta.json
        meta = {
            'slice_id': slice_index,
            'dataset': dataset_name,
            'start_time': slice_obj.start_time.isoformat(),
            'end_time': slice_obj.end_time.isoformat(),
            'num_records': int(len(slice_obj.df_slice))
        }
        meta_json_path = os.path.join(slice_dir, 'meta.json')
        with open(meta_json_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # Save full slice pickle for consumers that want Python objects
        pkl_path = os.path.join(slice_dir, 'slice.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(slice_obj, f)

        return slice_dir

    def save_slices(self, slices, dataset_name, prefix='slice'):
        """Save slices to data/processed/<dataset>/slicer/<slice_id> with CSV, meta, and pickle."""
        saved_dirs = []
        for i, s in enumerate(slices, start=1):
            saved_dir = self._save_single_slice(s, i, dataset_name)
            saved_dirs.append(saved_dir)
        return saved_dirs

    def process(self, filename, time_col='created_at', dataset_name='weibo', prefix='slice'):
        """Complete processing pipeline and save slices to processed directory structure."""
        df = self.load_data(filename)
        df = self.parse_time_column(df, time_col)
        slices = self.create_slices(df, time_col)
        self.save_slices(slices, dataset_name=dataset_name, prefix=prefix)
        return slices


def main():
    input_dir = os.path.join('data', 'raw')
    output_dir = os.path.join('data', 'processed')
    filename = 'weibo_posts.pkl'  # e.g., file under data/raw

    slicer = DataSlicer(input_dir, output_dir, interval_days=5, min_slice_size=100)
    slices = slicer.process(filename, time_col='created_at', dataset_name='weibo', prefix='weibo')

    # Print slice information
    for i, s in enumerate(slices):
        print(f"Slice {i + 1}: {s.start_time} -> {s.end_time}, Records: {len(s.df_slice)}")


if __name__ == '__main__':
    main()
