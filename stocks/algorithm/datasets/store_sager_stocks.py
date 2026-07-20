import math
import os

import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt
from stocks.algorithm.datasets import DATA_DIR


class StoreSagerStocks():
	NASDAQ_LIST_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
	OTHER_LIST_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

	def get_ticker_data(self, ticker, start_date, end_date, resolution="1d"):
		start_date_string = self._date_to_string(*start_date)
		end_date_string = self._date_to_string(*end_date)
		ticker = yf.Ticker(ticker, session=None)

		data = ticker.history(start=start_date_string, end=end_date_string, interval=resolution)
		data_pack = (data, ticker)
		return data_pack
	
	def get_data(self, start_date, end_date):
		files = os.listdir(str(DATA_DIR) + "/us_stocks")
		print(files)
		if not "us_ticker_lists.csv" in files:
			print("The US ticker list is not currently in your environment...")
			print("Downloading the US ticker list ;)")
			self._download_us_ticker_lists()

		tickers = pd.read_csv(str(DATA_DIR) + "/us_stocks/us_ticker_lists.csv")["symbol"].to_list()

		self._multi_scraping(str(DATA_DIR) + "/us_stocks/test/" , ["Volume", "Close"], tickers, start_date, end_date, resolution="1d")


	def _read_symbol_file(self, url: str, exchange_name: str, symbol_column: str) -> pd.DataFrame:
		data = pd.read_csv(url, sep="|", skipfooter=1, engine="python")
		data = data[data[symbol_column].notna()].copy()
		data["exchange"] = exchange_name
		return data

	def _download_us_ticker_lists(self, output_file: str = "data/us_stocks/us_ticker_lists.csv") -> pd.DataFrame:
		nasdaq = self._read_symbol_file(self.NASDAQ_LIST_URL, "NASDAQ", "Symbol")
		other = self._read_symbol_file(self.OTHER_LIST_URL, "NYSE/AMEX/ARCA", "ACT Symbol")

		nasdaq = nasdaq.rename(columns={"Symbol": "symbol", "Security Name": "name"})
		other = other.rename(columns={"ACT Symbol": "symbol", "Security Name": "name"})

		combined = pd.concat(
			[
				nasdaq[["symbol", "name", "exchange"]],
				other[["symbol", "name", "exchange"]],
			],
			ignore_index=True,
		)

		combined = combined.drop_duplicates(subset=["symbol", "exchange"]).sort_values(
			["exchange", "symbol"]
		)
		combined.to_csv(output_file, index=False)
		return combined

	def _train_val_split(self, path_to_folder, types, split_ratio, total_dataset_size, start_date, end_date):
		if isinstance(types, str):
			types = [types]

		value_files = np.array(os.listdir(path_to_folder + "/" + str(types[0]) + "_data"))

		train_data_frames = {}
		validation_data_frames = {}
		for data_type in types:
			train_data_frames[data_type] = pd.DataFrame()
			validation_data_frames[data_type] = pd.DataFrame()

		# randomize files
		shuffled_value_files = np.random.permutation(value_files)
		data_equiped = 0
		i = 0

		while data_equiped < total_dataset_size and i < len(shuffled_value_files):
			file_name = shuffled_value_files[i]
			current_data = {}
			for data_type in types:
				current_data[data_type] = pd.read_csv(path_to_folder + "/" + str(data_type) + "_data/" + file_name, index_col=0)

			# Shuffle once by ticker and keep that order for all data types.
			common_tickers = current_data[types[0]].index
			for data_type in types[1:]:
				common_tickers = common_tickers.intersection(current_data[data_type].index)

			shuffled_tickers = np.random.permutation(common_tickers)

			for ticker in shuffled_tickers:
				if data_equiped >= total_dataset_size:
					break

				for data_type in types:
					row = current_data[data_type].loc[ticker]
					train_row, validation_row = self.single_split(row, split_ratio, start_date, end_date)
					train_data_frames[data_type] = pd.concat([train_data_frames[data_type], train_row], axis=0, sort=False)
					validation_data_frames[data_type] = pd.concat([validation_data_frames[data_type], validation_row], axis=0, sort=False)

				data_equiped += 1

			i += 1

		# Create trainfolder with traning_data and validation_data
		try:
			os.makedirs("training", exist_ok=True)
			for data_type in types:
				train_data_frames[data_type] = train_data_frames[data_type].sort_index(axis=1)
				validation_data_frames[data_type] = validation_data_frames[data_type].sort_index(axis=1)
				train_data_frames[data_type].to_csv("training/" + str(data_type) + "_train.csv")
				validation_data_frames[data_type].to_csv("training/" + str(data_type) + "_val.csv")
		except:
			print("Could not create the training environment (training folder, training/train.csv, training/val.csv)")

		return train_data_frames, validation_data_frames

	def _single_split(self, row, split_ratio, start_date, end_date):
		row = pd.to_numeric(row, errors="coerce")
		row = row.dropna()

		row_index = pd.to_datetime(row.index)
		order = np.argsort(row_index)
		row = row.iloc[order]
		row_index = row_index[order]

		start_date = pd.to_datetime(self.date_to_string(*start_date))
		end_date = pd.to_datetime(self.date_to_string(*end_date))

		time_mask = (row_index >= start_date) & (row_index <= end_date)
		row = row.loc[time_mask]
		row_index = row_index[time_mask]

		split_point = int(len(row) * split_ratio)
		if split_point <= 0:
			split_point = 1
		if split_point >= len(row):
			split_point = len(row) - 1

		train_row = pd.DataFrame([row.iloc[:split_point].to_list()], index=[row.name], columns=row_index[:split_point])
		validation_row = pd.DataFrame([row.iloc[split_point:].to_list()], index=[row.name], columns=row_index[split_point:])

		train_row.index.name = "date"
		validation_row.index.name = "date"

		return train_row, validation_row

	def _single_plot(self, data):
		timestamps = data.index.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S%z").tolist()
		close = data["Close"].tolist()
		low = np.array(data["Low"].tolist())
		high = np.array(data["High"].tolist())

		fig, ax = plt.subplots()
		ax.plot(timestamps, close, label="Close", color="tab:blue")
		ax.fill_between(timestamps, low, high, color="tab:orange", alpha=0.2, label="Bound range")
		ax.legend(loc="best")

		n = len(timestamps)
		step = max(1, n // 10)
		ax.set_xticks(range(0, n, step))
		ax.set_xticklabels(timestamps[::step])

		angle = 45
		labels = ax.get_xticklabels()

		# Pass 1: measure each label's un-rotated width (needed to compute the correct shift)
		for label in labels:
			label.set_rotation(0)
			label.set_ha("right")
			label.set_va("top")
		fig.canvas.draw()
		renderer = fig.canvas.get_renderer()
		widths = [label.get_window_extent(renderer=renderer).width for label in labels]

		# Pass 2: rotate around the top-right corner (keeps the whole label below
		# the axis), then shift right by exactly half the text's horizontal
		# projection so the true center of the string lands under the tick
		for label, w in zip(labels, widths):
			label.set_rotation(angle)
			label.set_rotation_mode("anchor")
			dx_px = (w / 2) * math.cos(math.radians(angle))
			offset = mtransforms.ScaledTranslation(dx_px / fig.dpi, 0, fig.dpi_scale_trans)
			label.set_transform(label.get_transform() + offset)

		fig.tight_layout()
		plt.show()

	def _date_to_string(self, year, month, day):
		string_year = str(year)
		string_month = str(month)
		string_day = str(day)

		if len(string_month) == 1:
			string_month = "0" + string_month
		if len(string_day) == 1:
			string_day = "0" + string_day

		ymd = string_year + "-" + string_month + "-" + string_day

		return ymd

	def _multi_scraping(self, stock_data_file, data_type, tickers, start_date, end_date, resolution="1d"):
		length = len(tickers)
		i = 1
		n = 0
		multi_stock_data = pd.DataFrame()
		for ticker in tickers:
			print(str(i) + "/" + str(length))
			try:
				data_packet = self.get_ticker_data(ticker, start_date, end_date, resolution)
				
				# reformat
				single_stock_data = self._reformat_data(data_packet, data_type)
				multi_stock_data = pd.concat([multi_stock_data, single_stock_data], axis=0)
				multi_stock_data = multi_stock_data.reindex(sorted(multi_stock_data.columns), axis=1)
			except:
				print("Could not concatenate " + str(ticker) + " with the rest!")

			# save
			if i % 100 == 0:
				multi_stock_data.to_csv(stock_data_file + str(n) + ".csv")
				multi_stock_data = pd.DataFrame()
				n += 1

			i += 1

		# Flush any remaining tickers that did not complete a full batch of 100.
		if not multi_stock_data.empty:
			multi_stock_data.to_csv(stock_data_file + str(n) + ".csv")

	def _reformat_data(self, data_pack, type):
		data = data_pack[0]
		tick = data_pack[1]
		time_stamps = np.array(data.index.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S%z").tolist())
		time_stamps = np.array([x[:10] for x in time_stamps.astype(str)])
		closing_price = np.array(data[type].to_list()).astype(np.float32)

		# create pandas data frame
		data_frame = pd.DataFrame([closing_price], index=[tick], columns=time_stamps)
		data_frame.index.name = "date"
		print(data_frame)
		return data_frame
    
if __name__ == "__main__":
	dataSet = StoreSagerStocks()

	dataSet.get_data((2018, 1, 1), (2026, 7, 20))