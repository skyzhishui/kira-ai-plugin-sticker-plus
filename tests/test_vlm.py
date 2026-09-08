"""VLM adapter tests: JSON parsing, selection index handling, tagging."""

from __future__ import annotations

import pytest

from helpers import EmojiVLM, FakeClient, make_png_bytes


def make_vlm(reply: str) -> tuple[EmojiVLM, FakeClient]:
    client = FakeClient([reply])
    return EmojiVLM(client_resolver=lambda: client), client


# ----------------------------------------------------------------------
# JSON parsing
# ----------------------------------------------------------------------

def test_parse_plain_json():
    assert EmojiVLM._parse_json_object('{"a": 1}') == {"a": 1}


def test_parse_fenced_json():
    raw = '```json\n{"a": 1}\n```'
    assert EmojiVLM._parse_json_object(raw) == {"a": 1}


def test_parse_json_with_prose():
    raw = '好的，结果如下：{"emoji_index": 2, "reason": "合适"} 以上。'
    assert EmojiVLM._parse_json_object(raw)["emoji_index"] == 2


def test_parse_invalid_raises():
    with pytest.raises(RuntimeError):
        EmojiVLM._parse_json_object("完全没有 JSON")


def test_fallback_extract_selection():
    raw = '{"emoji_index": 3, "reason": "它带着「开心」的表情"}'
    parsed = EmojiVLM._fallback_extract_emoji_selection(raw)
    assert parsed == {"emoji_index": 3, "reason": "它带着「开心」的表情"}


def test_fallback_extract_unescaped_quotes():
    raw = '{"emoji_index": 2, "reason": "这是"最佳"选择"}'
    parsed = EmojiVLM._fallback_extract_emoji_selection(raw)
    assert parsed["emoji_index"] == 2
    assert "最佳" in parsed["reason"]


def test_fallback_extract_failure_returns_none():
    assert EmojiVLM._fallback_extract_emoji_selection("no index here") is None


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_select_returns_zero_based_index():
    vlm, _ = make_vlm('{"emoji_index": 2, "reason": "匹配"}')
    idx, reason = await vlm.select_emoji_by_description(["a", "b", "c"], "开心")
    assert idx == 1 and reason == "匹配"


@pytest.mark.asyncio
async def test_select_out_of_range_raises():
    vlm, _ = make_vlm('{"emoji_index": 9, "reason": "x"}')
    with pytest.raises(RuntimeError):
        await vlm.select_emoji_by_description(["a", "b"], "开心")


@pytest.mark.asyncio
async def test_select_broken_json_rescued():
    vlm, _ = make_vlm('{"emoji_index": 1, "reason": "最"贴切"的一张"}')
    idx, _ = await vlm.select_emoji_by_description(["a", "b"], "开心")
    assert idx == 0


@pytest.mark.asyncio
async def test_select_empty_candidates_raises():
    vlm, _ = make_vlm("{}")
    with pytest.raises(ValueError):
        await vlm.select_emoji_by_description([], "开心")


@pytest.mark.asyncio
async def test_selection_routes_through_dedicated_client():
    # Selection is text-only and may use a different (non-vision) model.
    tag_client = FakeClient([])
    sel_client = FakeClient(['{"emoji_index": 1, "reason": "匹配"}'])
    vlm = EmojiVLM(
        client_resolver=lambda: tag_client,
        selection_client_resolver=lambda: sel_client,
    )
    idx, _ = await vlm.select_emoji_by_description(["a", "b"], "开心")
    assert idx == 0
    assert len(sel_client.requests) == 1
    assert not tag_client.requests, "selection must not touch the tagging client"
    content = sel_client.requests[0].messages[0]["content"]
    assert [part["type"] for part in content] == ["text"]


@pytest.mark.asyncio
async def test_selection_falls_back_to_tagging_client():
    client = FakeClient(['{"emoji_index": 1, "reason": "匹配"}'])
    vlm = EmojiVLM(client_resolver=lambda: client)
    _, _ = await vlm.select_emoji_by_description(["a"], "开心")
    assert len(client.requests) == 1


def test_selection_client_none_raises():
    vlm = EmojiVLM(
        client_resolver=lambda: object(),
        selection_client_resolver=lambda: None,
    )
    with pytest.raises(RuntimeError):
        vlm._selection_client()


# ----------------------------------------------------------------------
# Tagging
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tag_emoji_parses_result(tmp_path):
    png = make_png_bytes()
    file_path = tmp_path / "x.png"
    file_path.write_bytes(png)
    vlm, client = make_vlm('{"description": "一只猫在笑", "emotions": "开心,调皮"}')
    description, emotions = await vlm.tag_emoji(file_path)
    assert description == "一只猫在笑"
    assert emotions == "开心,调皮"
    # one request, multimodal content with image_url + text
    assert len(client.requests) == 1
    content = client.requests[0].messages[0]["content"]
    types = [part["type"] for part in content]
    assert types == ["image_url", "text"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_tag_emoji_missing_fields_raise(tmp_path):
    file_path = tmp_path / "x.png"
    file_path.write_bytes(make_png_bytes())
    vlm, _ = make_vlm('{"description": "只有描述"}')
    with pytest.raises(RuntimeError):
        await vlm.tag_emoji(file_path)


def test_client_resolver_none_raises():
    vlm = EmojiVLM(client_resolver=lambda: None)
    with pytest.raises(RuntimeError):
        vlm._client()
