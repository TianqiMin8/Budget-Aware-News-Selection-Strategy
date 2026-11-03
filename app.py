"""Local Streamlit demo for the news-to-prediction workflow."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAIError

from newsalpha.events import OpenAIEventExtractor
from newsalpha.ingestion import load_stock_prices_archive
from newsalpha.pipeline import (
    TRAIN_END,
    VALIDATION_END,
    build_market_features,
    run_ticker_demo,
)

st.set_page_config(page_title="NewsAlpha Demo", layout="wide")

ROOT = Path(__file__).resolve().parent
NEWS_ARCHIVE = ROOT / "data" / "financial_news_project.zip"
STOCK_ARCHIVE = ROOT / "data" / "stock_prices.zip"
TICKERS = (
    "AAPL", "ABBV", "AMD", "AMZN", "BA", "CAT", "DIS",
    "GS", "MS", "MSFT", "NKE", "NVDA", "TSLA", "V",
)
METHODS = {
    "Recency": "recency",
    "Relevance": "relevance",
    "MMR": "mmr",
    "Novelty / coverage (initial)": "novelty_coverage",
}


@st.cache_data(show_spinner="Loading archived market data...")
def _market_features(path: str) -> pd.DataFrame:
    return build_market_features(load_stock_prices_archive(path))


def _show_metrics(title: str, metrics: dict[str, float | int]) -> None:
    st.subheader(title)
    columns = st.columns(4)
    columns[0].metric("MAE", f"{metrics['mae'] * 100:.3f}%")
    columns[1].metric("RMSE", f"{metrics['rmse'] * 100:.3f}%")
    columns[2].metric("Direction accuracy", f"{metrics['directional_accuracy']:.2%}")
    columns[3].metric("Validation rows", str(metrics["n_observations"]))


def main() -> None:
    st.title("NewsAlpha: 新闻选择 → 事件抽取 → 收益预测")
    st.caption("本机 Streamlit 页面 | 历史新闻与股价 | 默认 Mock 模式，不调用外部 API")

    if not NEWS_ARCHIVE.exists() or not STOCK_ARCHIVE.exists():
        st.error("未找到项目数据压缩包。请从仓库根目录运行：streamlit run app.py")
        st.stop()

    try:
        market = _market_features(str(STOCK_ARCHIVE))
    except (OSError, ValueError) as error:
        st.error(f"读取股价数据失败：{error}")
        st.stop()

    ticker = st.sidebar.selectbox("Ticker", TICKERS)
    dates = sorted(
        market.loc[
            (market["ticker"] == ticker)
            & (market["date"] >= TRAIN_END)
            & (market["date"] < VALIDATION_END),
            "date",
        ].unique()
    )
    if not dates:
        st.error(f"{ticker} 没有可用的 Validation 日期。")
        st.stop()
    selected_date = st.sidebar.selectbox(
        "预测日期（Validation）",
        dates,
        index=len(dates) - 1,
        format_func=lambda value: value.isoformat(),
    )
    method_label = st.sidebar.selectbox("新闻选择方法", tuple(METHODS))
    token_budget = st.sidebar.slider(
        "每日新闻 token 预算",
        min_value=250,
        max_value=5000,
        value=1500,
        step=250,
    )
    st.sidebar.caption(
        "新闻按美东时间 16:00 截止，避免使用收盘后信息。"
    )

    run_clicked = st.button("运行端到端 Demo", type="primary")
    if run_clicked:
        with st.spinner("读取公司新闻、选择文章、Mock 抽取并训练 Ridge..."):
            try:
                st.session_state["demo_result"] = run_ticker_demo(
                    NEWS_ARCHIVE,
                    STOCK_ARCHIVE,
                    ticker,
                    selected_date.isoformat(),
                    METHODS[method_label],
                    token_budget,
                )
                st.session_state["demo_parameters"] = (
                    ticker,
                    selected_date.isoformat(),
                    METHODS[method_label],
                    token_budget,
                )
            except (OSError, ValueError) as error:
                st.error(f"Demo 未能完成：{error}")
                st.session_state.pop("demo_result", None)
                st.session_state.pop("demo_parameters", None)

    parameters = (ticker, selected_date.isoformat(), METHODS[method_label], token_budget)
    result = st.session_state.get("demo_result")
    if result is None or st.session_state.get("demo_parameters") != parameters:
        st.info("选择参数后点击“运行端到端 Demo”。")
        return

    st.success(
        f"{result.ticker} · {result.prediction_date} · {method_label} · "
        f"{len(result.selected_articles)} 篇文章"
    )
    st.info(
        "离线演示使用本地关键词 Mock 抽取器；验证集指标是单一股票、"
        "Mock 新闻特征的结果，不等同于全 14 家公司的 Baseline 或真实 LLM 实验。"
    )

    baseline_col, news_col = st.columns(2)
    with baseline_col:
        _show_metrics("量价 Baseline", result.baseline_metrics)
    with news_col:
        _show_metrics("量价 + 新闻事件（Mock）", result.news_metrics)

    prediction_columns = st.columns(3)
    prediction_columns[0].metric(
        "Baseline 预测下一交易日收益",
        f"{result.baseline_prediction * 100:.3f}%",
    )
    prediction_columns[1].metric(
        "加入新闻后的预测",
        f"{result.news_prediction * 100:.3f}%",
    )
    prediction_columns[2].metric(
        "历史实际收益（仅作回看）",
        f"{result.actual_return * 100:.3f}%",
    )

    used_tokens = int(result.selected_articles["token_count_used"].sum())
    st.subheader("本次选择与抽取")
    st.write(
        f"当前日期选中 **{len(result.selected_articles)}** 篇，"
        f"估算 **{used_tokens}/{token_budget} tokens**；"
        f"Train + Validation 历史抽取输入约 {result.mock_usage.input_tokens:,} tokens。"
    )
    st.caption(
        f"Mock 抽取器：{result.mock_usage.model_name}；API 调用：0；"
        "实际 API 费用：$0.00。Mock token 数仅为成本 dry-run 的估算，不是计费记录。"
    )
    if result.selected_articles.empty:
        st.warning("所选日期没有符合预算的新闻。")
    else:
        display_columns = [
            "selection_rank",
            "article_id",
            "published_at",
            "title",
            "selection_score",
            "token_count_used",
        ]
        st.dataframe(
            result.selected_articles[display_columns],
            hide_index=True,
        )
        st.download_button(
            "下载当前日期选中的文章",
            result.selected_articles.to_csv(index=False).encode("utf-8"),
            file_name=f"{result.ticker}_{result.prediction_date}_selected.csv",
            mime="text/csv",
        )

    if not result.extracted_events.empty:
        st.subheader("当前日期事件")
        st.dataframe(result.extracted_events, hide_index=True)

    st.divider()
    st.subheader("可选：真实 OpenAI 抽取预览")
    st.warning(
        "该预览会产生真实 API 调用及费用，最多发送当前选择结果中的 5 篇文章；"
        "结果只作抽取对照，不会用于上方 Ridge 预测。"
    )
    consent = st.checkbox("我确认要发送上述文章到 OpenAI，并承担相应费用")
    if st.button("发送 OpenAI 预览请求", disabled=not consent):
        if not os.getenv("OPENAI_API_KEY"):
            st.error("请先在环境变量中设置 OPENAI_API_KEY。")
        elif result.selected_articles.empty:
            st.error("当前日期没有可发送的已选文章。")
        else:
            sample = result.selected_articles.head(5)
            articles = [
                {
                    "article_id": str(row.article_id),
                    "title": str(row.title),
                    "text": str(row.text),
                }
                for row in sample.itertuples(index=False)
            ]
            try:
                preview = OpenAIEventExtractor().extract(articles)
            except (ImportError, OpenAIError, ValueError) as error:
                st.error(f"OpenAI 抽取失败：{error}")
            else:
                st.write(
                    f"模型：{preview.model_name}；输入 {preview.input_tokens:,} tokens；"
                    f"输出 {preview.output_tokens:,} tokens；"
                    f"响应时间 {preview.latency_seconds:.2f} 秒。"
                )
                st.caption(
                    "API 响应未提供美元费用；请以 OpenAI 账户用量/账单为准。"
                )
                st.json(
                    [
                        {
                            "article_id": event.article_id,
                            "event_type": event.event_type,
                            "description": event.description,
                            "impact_direction": event.impact_direction,
                            "confidence": event.confidence,
                            "evidence_quote": event.evidence_quote,
                        }
                        for event in preview.events
                    ]
                )


if __name__ == "__main__":
    main()
