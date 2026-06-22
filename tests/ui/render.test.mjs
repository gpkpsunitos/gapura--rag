import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const source = fs.readFileSync(new URL('../../static/app.js', import.meta.url), 'utf8');
const context = { console, globalThis: {} };
context.global = context.globalThis;

vm.runInNewContext(source, context);

const {
    buildEvidenceHtml,
    buildGroundingMetaHtml,
    formatMessageHtml,
    normalizeAssistantText,
} = context.globalThis.__appTest;

test('buildEvidenceHtml renders evidence ids and snippets', () => {
    const html = buildEvidenceHtml({
        language: 'en',
        evidence: [
            {
                id: 'E1',
                source: 'manual.pdf',
                page: 4,
                snippet: 'Baggage counter opens at 05:00.',
            },
        ],
    });

    assert.match(html, /E1/);
    assert.match(html, /manual\.pdf/);
    assert.match(html, /Baggage counter opens at 05:00/);
});

test('buildGroundingMetaHtml surfaces partial grounding state', () => {
    const html = buildGroundingMetaHtml({
        language: 'en',
        grounding_status: 'partial',
        supplement_used: true,
    });

    assert.match(html, /Partially grounded/);
    assert.match(html, /outside-document supplement/);
});

test('formatMessageHtml keeps citations visible across paragraphs', () => {
    const html = formatMessageHtml('Line one [E1].\n\nLine two [E2].');
    assert.match(html, /<p>Line one \[E1\]\.<\/p><p>Line two \[E2\]\.<\/p>/);
});

test('normalizeAssistantText extracts answer from raw JSON payloads', () => {
    const normalized = normalizeAssistantText(
        '{"grounding_status":"grounded","answer":"Jawaban final [E1].","cited_evidence_ids":["E1"],"supplement":null}',
        'id',
    );

    assert.equal(normalized.text, 'Jawaban final [E1].');
});
