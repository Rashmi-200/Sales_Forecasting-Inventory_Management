import os
import re
import joblib
import requests
import math
import json
import argparse
import datetime
import csv
from collections import Counter
from dotenv import load_dotenv

# Load environment
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
NEWS_API_KEY = os.getenv('NEWS_API_KEY')


def load_model_predictor():
    # import prediction function from your training script
    try:
        # Try package-aware absolute import first (when executed as module)
        from backend.train_customer_segmentation import predict_customer_segment
    except Exception:
        try:
            # Fallback to local module import for direct script runs
            from train_customer_segmentation import predict_customer_segment
        except Exception:
            raise RuntimeError('Could not import predict_customer_segment from train_customer_segmentation.py')
    return predict_customer_segment


PLATFORMS = ['Facebook', 'Instagram', 'YouTube', 'TikTok', 'Twitter', 'LinkedIn', 'Pinterest', 'Reddit']


def newsapi_search(query, page_size=50):
    if not NEWS_API_KEY:
        return []
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': query,
        'pageSize': page_size,
        'language': 'en',
        'sortBy': 'relevancy',
        'apiKey': NEWS_API_KEY
    }
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200:
        return []
    data = resp.json()
    articles = data.get('articles', [])
    texts = []
    for a in articles:
        texts.append(' '.join(filter(None, [a.get('title',''), a.get('description','')])) )
    return texts


def score_platforms_by_news(age_label):
    # Query NewsAPI for social media trends for the age group
    query = f"social media trends {age_label} demographics OR "
    query += ' OR '.join(PLATFORMS)
    texts = newsapi_search(query)
    counter = Counter()
    if not texts:
        # fallback heuristic mapping
        return heuristic_platforms(age_label)

    pattern = re.compile(r'\b(' + '|'.join([p for p in PLATFORMS]) + r')\b', flags=re.I)
    for t in texts:
        for m in pattern.findall(t):
            counter[m.title()] += 1

    # If no mentions, fallback
    if not counter:
        return heuristic_platforms(age_label)

    # Rank platforms by count
    ranked = [p for p, _ in counter.most_common()]
    # Append any platforms not seen
    for p in PLATFORMS:
        if p not in ranked:
            ranked.append(p)
    return ranked


def heuristic_platforms(age_label):
    # Simple heuristics
    if '0-24' in age_label or '25-34' in age_label:
        return ['TikTok', 'Instagram', 'YouTube', 'Twitter', 'Facebook', 'Reddit', 'Pinterest', 'LinkedIn']
    if '35-44' in age_label or '45-59' in age_label:
        return ['Facebook', 'YouTube', 'Instagram', 'LinkedIn', 'Twitter', 'Pinterest', 'Reddit', 'TikTok']
    return ['Facebook', 'YouTube', 'Instagram', 'LinkedIn', 'Twitter', 'Pinterest', 'Reddit', 'TikTok']


CPM_USD = {
    'Facebook': 7.0,
    'Instagram': 6.0,
    'YouTube': 9.0,
    'TikTok': 5.0,
    'Twitter': 6.0,
    'LinkedIn': 12.0,
    'Pinterest': 5.0,
    'Reddit': 4.0
}


def estimate_costs(platforms, impressions_by_platform=None):
    # default impressions per platform
    if impressions_by_platform is None:
        impressions_by_platform = {p: 10000 for p in platforms}

    costs = {}
    for p in platforms:
        imp = impressions_by_platform.get(p, 10000)
        cpm = CPM_USD.get(p, 6.0)
        cost = (imp / 1000.0) * cpm
        costs[p] = round(cost, 2)
    total = round(sum(costs.values()), 2)
    return costs, total


def write_report_csv(output, cpms_override=None, impressions_override=None, out_dir=None):
    out_dir = out_dir or os.path.dirname(__file__)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(out_dir, f'campaign_report_{ts}.csv')
    json_path = os.path.join(out_dir, f'campaign_report_{ts}.json')

    ranked = output['recommended_platforms_ranked']
    costs = output['estimated_costs_usd']

    # Compose CSV rows
    rows = []
    for rank, p in enumerate(ranked, start=1):
        impressions = None
        if impressions_override and p in impressions_override:
            impressions = impressions_override[p]
        else:
            impressions = 10000
        cpm = None
        if cpms_override and p in cpms_override:
            cpm = cpms_override[p]
        else:
            cpm = CPM_USD.get(p, None)
        cost = costs.get(p, None)
        rows.append({'Platform': p, 'Rank': rank, 'Impressions': impressions, 'CPM_USD': cpm, 'Cost_USD': cost})

    # Write CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Rank', 'Platform', 'Impressions', 'CPM_USD', 'Cost_USD'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Write JSON full output
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    return csv_path, json_path


def campaign_advice(segment_label, top_platforms):
    adv = []
    # Short messaging tips
    adv.append(f"Target messaging: emphasize product benefits for {segment_label.split('_')[0]} age group and {segment_label.split('_')[-1]} customers.")
    adv.append('Creative: use short demo videos and user testimonials for wearables like smartwatches.')
    adv.append('Timing: schedule posts during evenings and weekends to maximize engagement for working adults.')
    adv.append('CTA: highlight trial, limited-time discount, and easy returns to reduce purchase friction.')
    adv.append('Audiences: use lookalike audiences on Facebook/Instagram and interest-based targeting on YouTube.')
    # platform-specific quick tips
    tips = {
        'Facebook': 'Use carousel ads + dynamic product ads; strong for 35-59 age segment.',
        'Instagram': 'Focus on Reels + influencer partnerships for visual product demos.',
        'YouTube': 'Create 15–30s pre-roll demo videos plus product review collaborations.',
        'TikTok': 'Short creative challenges and influencer partnerships (younger skew).',
        'LinkedIn': 'Use for professional positioning and B2B-style messaging (higher CPM).',
        'Reddit': 'Engage in relevant subreddits for authentic product discussion (use caution).'
    }
    platform_bullets = [f"{p}: {tips.get(p, 'Use targeted creative and measure performance quickly.')}" for p in top_platforms[:4]]
    return adv + platform_bullets


def plan_for_input(product_name, brand, category, impressions_by_platform=None):
    predictor = load_model_predictor()
    seg = predictor(product_name=product_name, brand=brand, category=category)
    # seg example: {'Age_Group': '36–45', 'Gender': 'Male'}
    age_group = seg['Age_Group']
    gender = seg['Gender']
    segment_label = f"{age_group}_{gender}"

    ranked = score_platforms_by_news(age_group)
    costs, total = estimate_costs(ranked, impressions_by_platform)
    advice = campaign_advice(segment_label, ranked)

    output = {
        'predicted_segment': seg,
        'recommended_platforms_ranked': ranked,
        'estimated_costs_usd': costs,
        'total_estimated_cost_usd': total,
        'advice': advice
    }
    return output


def parse_kv_str(s: str):
    """Parse key=value pairs or JSON string into dict."""
    if not s:
        return None
    s = s.strip()
    # try JSON first
    try:
        return json.loads(s)
    except Exception:
        pass
    # try semicolon or comma separated key=value
    out = {}
    for part in re.split(r'[;,]\s*', s):
        if not part:
            continue
        if '=' in part:
            k, v = part.split('=', 1)
            k = k.strip()
            v = v.strip()
            try:
                vnum = float(v)
                if vnum.is_integer():
                    v = int(vnum)
                else:
                    v = vnum
            except Exception:
                pass
            out[k] = v
    return out


def main():
    p = argparse.ArgumentParser(description='Plan a campaign for a product using customer segmentation')
    p.add_argument('--product', required=False, default='Smartwatch')
    p.add_argument('--brand', required=False, default='Sony')
    p.add_argument('--category', required=False, default='Electronics')
    p.add_argument('--impressions', required=False, help='Impressions as JSON string or key=value list, e.g. "Facebook=20000,Twitter=10000"')
    p.add_argument('--cpms', required=False, help='CPMs as JSON string or key=value list, e.g. "Facebook=7,Twitter=6"')
    p.add_argument('--impressions-file', required=False, help='Path to JSON file with impressions dict')
    p.add_argument('--cpms-file', required=False, help='Path to JSON file with CPMs dict')
    p.add_argument('--output-dir', required=False, default=os.path.dirname(__file__))
    args = p.parse_args()

    impressions_override = None
    cpms_override = None
    # load from files if provided
    if args.impressions_file:
        try:
            with open(args.impressions_file, 'r', encoding='utf-8') as fh:
                impressions_override = json.load(fh)
        except Exception as e:
            print('Could not read impressions file:', e)
    if args.cpms_file:
        try:
            with open(args.cpms_file, 'r', encoding='utf-8') as fh:
                cpms_override = json.load(fh)
        except Exception as e:
            print('Could not read cpms file:', e)

    # parse inline strings if provided
    if args.impressions:
        impressions_override = parse_kv_str(args.impressions)
    if args.cpms:
        cpms_override = parse_kv_str(args.cpms)

    # apply cpms_override to global CPM_USD for computation
    global CPM_USD
    if cpms_override:
        CPM_USD = {**CPM_USD, **{k: float(v) for k, v in cpms_override.items()}}

    out = plan_for_input(args.product, args.brand, args.category, impressions_by_platform=impressions_override)

    csv_path, json_path = write_report_csv(out, cpms_override=cpms_override, impressions_override=impressions_override, out_dir=args.output_dir)

    print('\n🎯 Sample Campaign Plan:')
    print(json.dumps(out, indent=2))
    print('\nSaved CSV report to:', csv_path)
    print('Saved JSON report to:', json_path)


if __name__ == '__main__':
    main()
