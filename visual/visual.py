# -*- coding: utf-8 -*-

import pandas as pd
import plotly.graph_objects as go

class Visual:
    def __init__(self):
        pass

    def plot_price_and_forecasts1(self, df: pd.DataFrame):
        fig = go.Figure()

        # Рисуем фактическую цену (столбец 'praice' — исправь, если иначе называется)
        fig.add_trace(go.Scatter(
            x=df['data'],
            y=df['praice'],
            mode='lines+markers',
            name='Price',
            line=dict(color='blue')
        ))

        df=df.drop(columns=["praice"])
        for col in df:
            mask = df[col].notnull()
            fig.add_trace(go.Scatter(
                x=df['data'][mask],
                y=df[col][mask],
                mode='lines+markers',
                name=col,
                line=dict(dash='dot', width=2),
                marker=dict(size=6)
            ))

        fig.update_layout(
            title='Price and Forecasts',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_white',
            hovermode='x unified'
        )

        fig.show()

    def plot_price_and_forecasts2(self, df: pd.DataFrame):
    

        fig = go.Figure()

        # Рисуем фактическую цену
        fig.add_trace(go.Scatter(
            x=df['data'],
            y=df['praice'],
            mode='lines+markers',
            name='Price',
            line=dict(color='blue')
        ))

        # Выделим группы колонок
        model_cols_30min = [col for col in df.columns if '_30min' in col]
        model_cols_1hour = [col for col in df.columns if '_1hour' in col]
        model_cols_4hour = [col for col in df.columns if '_4hour' in col]

        # Убедимся, что все числовые
        for cols in [model_cols_30min, model_cols_1hour, model_cols_4hour]:
            df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')


        # Добавим средние по группам
        if model_cols_30min:
            df['average_30min'] = df[model_cols_30min].mean(axis=1)
            mask = df['average_30min'].notnull()
            fig.add_trace(go.Scatter(
                x=df['data'][mask],
                y=df['average_30min'][mask],
                mode='lines+markers',
                name='Avg 30min',
                line=dict(color='green', width=3)
            ))

        if model_cols_1hour:
            df['average_1hour'] = df[model_cols_1hour].mean(axis=1)
            mask = df['average_1hour'].notnull()
            fig.add_trace(go.Scatter(
                x=df['data'][mask],
                y=df['average_1hour'][mask],
                mode='lines+markers',
                name='Avg 1hour',
                line=dict(color='orange', width=3, dash='dash')
            ))

        if model_cols_4hour:
            df['average_4hour'] = df[model_cols_4hour].mean(axis=1)
            mask = df['average_4hour'].notnull()
            fig.add_trace(go.Scatter(
                x=df['data'][mask],
                y=df['average_4hour'][mask],
                mode='lines+markers',
                name='Avg 4hour',
                line=dict(color='red', width=3, dash='dot')
            ))

        # Удалим 'praice' и временные average, чтобы не рисовать их повторно
        exclude_cols = ['praice', 'average_30min', 'average_1hour', 'average_4hour']
        plot_cols = [col for col in df.columns if col not in exclude_cols and col != 'data']

        # Добавим отдельные прогнозы
        for col in plot_cols:
            mask = df[col].notnull()
            fig.add_trace(go.Scatter(
                x=df['data'][mask],
                y=df[col][mask],
                mode='lines+markers',
                name=col,
                line=dict(dash='dot', width=1),
                marker=dict(size=4)
            ))

        fig.update_layout(
            title='Price and Forecasts',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_white',
            hovermode='x unified'
        )

        fig.write_html("./out/report.html")

    def plot_price_and_forecasts(self, df: pd.DataFrame):
        # import plotly.graph_objects as go

        df['data'] = pd.to_datetime(df['data'])
        df.set_index('data', inplace=True)
        df = df[~df.index.duplicated(keep='first')]

        model_cols_30min = [col for col in df.columns if '_30min' in col]
        model_cols_1hour = [col for col in df.columns if '_1hour' in col]
        model_cols_4hour = [col for col in df.columns if '_4hour' in col]
        all_model_cols = model_cols_30min + model_cols_1hour + model_cols_4hour

        df[all_model_cols] = df[all_model_cols].apply(pd.to_numeric, errors='coerce')

        df_resampled = df.resample('5T').mean()

        fig = go.Figure()

        # Фактическая цена
        if 'praice' in df.columns:
            df_price = df[['praice']].resample('5T').mean()
            fig.add_trace(go.Scatter(
                x=df_price.index,
                y=df_price['praice'],
                mode='lines+markers',
                name='Price',
                line=dict(color='blue'),
                connectgaps=True
            ))

        # Средние
        if model_cols_30min:
            df_resampled['average_30min'] = df_resampled[model_cols_30min].mean(axis=1)
            fig.add_trace(go.Scatter(
                x=df_resampled.index,
                y=df_resampled['average_30min'],
                mode='lines+markers',
                name='Avg 30min',
                line=dict(color='green', width=3),
                connectgaps=True
            ))

        if model_cols_1hour:
            df_resampled['average_1hour'] = df_resampled[model_cols_1hour].mean(axis=1)
            fig.add_trace(go.Scatter(
                x=df_resampled.index,
                y=df_resampled['average_1hour'],
                mode='lines+markers',
                name='Avg 1hour',
                line=dict(color='orange', width=3, dash='dash'),
                connectgaps=True
            ))

        if model_cols_4hour:
            df_resampled['average_4hour'] = df_resampled[model_cols_4hour].mean(axis=1)
            fig.add_trace(go.Scatter(
                x=df_resampled.index,
                y=df_resampled['average_4hour'],
                mode='lines+markers',
                name='Avg 4hour',
                line=dict(color='red', width=3, dash='dot'),
                connectgaps=True
            ))

        if all_model_cols:
            df_resampled['average_all'] = df_resampled[all_model_cols].mean(axis=1)
            fig.add_trace(go.Scatter(
                x=df_resampled.index,
                y=df_resampled['average_all'],
                mode='lines+markers',
                name='Avg All Models',
                line=dict(color='black', width=4, dash='dashdot'),
                connectgaps=True
            ))

        # Отдельные модели — полупрозрачные
        for col in all_model_cols:
            mask = df[col].notnull()
            fig.add_trace(go.Scatter(
                x=df.index[mask],
                y=df[col][mask],
                mode='lines+markers',
                name=col,
                line=dict(width=1, color='gray'),
                opacity=0.2,
                showlegend=False,  # не отображаем в легенде
                connectgaps=True
            ))

        fig.update_layout(
            title='Price and Forecasts (Averaged)',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_white',
            hovermode='x unified'
        )

        fig.show()

