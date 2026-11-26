from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jev_billing_tokenizer import JevBillingTokenizer

OPENROUTER_URL = "https://openrouter.ai/api/alpha/decisions"

WORKLOADS: list[tuple[str, str, str]] = [
    (
        "Thai",
        "non_latin_no_spaces",
        "ปัญญาประดิษฐ์และการเรียนรู้ของเครื่องถูกนำมาใช้อย่างแพร่หลายในการจำแนกเอกสารอัตโนมัติ การวิเคราะห์ข้อความลูกค้า และการประมวลผลภาษาธรรมชาติ",
    ),
    (
        "Russian (Cyrillic)",
        "non_latin_inflected",
        "Искусственный интеллект и машинное обучение активно применяются для автоматической классификации документов, анализа пользовательских обращений и обработки естественного языка в корпоративных системах.",
    ),
    (
        "Ukrainian (Cyrillic)",
        "non_latin_inflected",
        "Штучний інтелект та машинне навчання використовуються для автоматичної класифікації звернень користувачів, перевірки фінансових операцій і модерації контенту в реальному часі.",
    ),
    (
        "Arabic",
        "non_latin_rtl",
        "يُستخدم الذكاء الاصطناعي وتعلم الآلة على نطاق واسع في تصنيف المستندات وتحليل رسائل العملاء ومعالجة اللغة الطبيعية في الأنظمة المؤسسية الحديثة.",
    ),
    (
        "Greek",
        "non_latin_inflected",
        "Η τεχνητή νοημοσύνη και η μηχανική μάθηση χρησιμοποιούνται ευρέως για την αυτόματη ταξινόμηση εγγράφων και την ανάλυση κειμένου σε σύγχρονες εφαρμογές.",
    ),
    (
        "Objective-C / Apple SDK",
        "camelcase_code",
        """@implementation UIViewControllerTransitionCoordinatorHelper
- (void)applicationDidFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
    NSManagedObjectContext *managedObjectContext = self.persistentContainer.viewContext;
    NSURLSessionConfiguration *sessionConfiguration = [NSURLSessionConfiguration defaultSessionConfiguration];
    NSAttributedString *attributedTitle = [[NSAttributedString alloc] initWithString:@"NavigationBarTitle" attributes:@{NSForegroundColorAttributeName: UIColor.systemBackgroundColor}];
    [[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(managedObjectContextDidSaveNotification:) name:NSManagedObjectContextDidSaveNotification object:managedObjectContext];
}""",
    ),
    (
        "Chinese (Simplified)",
        "non_latin_no_spaces",
        "人工智能和机器学习广泛应用于文档自动分类、客户工单分流、金融风控审核以及企业级实时自然语言处理系统中。",
    ),
    (
        "Japanese (Kanji+Kana)",
        "non_latin_no_spaces",
        "人工知能と機械学習は、文書の自動分類、顧客からの問い合わせ分析、不正検知、リアルタイムの自然言語処理システムで広く活用されています。",
    ),
    (
        "Hebrew",
        "non_latin_rtl",
        "בינה מלאכותית ולמידת מכונה משמשות באופן נרחב לסיווג אוטומטי של מסמכים, ניתוח פניות לקוחות ועיבוד שפה טבעית במערכות ארגוניות מתקדמות.",
    ),
    (
        "Android / Java (Spring + UI)",
        "camelcase_code",
        """public class MainActivity extends AppCompatActivity implements NavigationView.OnNavigationItemSelectedListener {
    private RecyclerView recyclerView;
    private ConstraintLayout constraintLayout;
    private SharedPreferences sharedPreferences;
    private AbstractSingletonProxyFactoryBean proxyFactoryBean;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ViewModelProvider viewModelProvider = new ViewModelProvider(this);
        HttpServletRequest servletRequest = RequestContextHolder.currentRequestAttributes();
    }
}""",
    ),
    (
        "Hindi (Devanagari)",
        "non_latin_indic",
        "कृत्रिम बुद्धिमत्ता और मशीन लर्निंग का उपयोग दस्तावेजों के स्वचालित वर्गीकरण, ग्राहक संदेशों के विश्लेषण और प्राकृतिक भाषा प्रसंस्करण के लिए व्यापक रूप से किया जाता है।",
    ),
    (
        "Polish",
        "latin_diacritics",
        "Zażółć gęślą jaźń. Przedsiębiorstwo telekomunikacyjne wprowadziło nowe rozporządzenie dotyczące bezpieczeństwa przetwarzania danych osobowych użytkowników.",
    ),
    (
        "German (compound nouns)",
        "latin_compounds",
        "Die Datenschutzgrundverordnung verpflichtet Versicherungsgesellschaften zur unverzüglichen Benachrichtigung der Aufsichtsbehörden bei Sicherheitsvorfällen. Kraftfahrzeughaftpflichtversicherung, Geschwindigkeitsbegrenzung, Krankenversicherungsbeiträge, Betriebsverfassungsgesetz, Einkommensteuererklärung.",
    ),
    (
        "TypeScript / React",
        "camelcase_code",
        """export const UserProfileDashboardContainer: React.FC<UserProfileDashboardProps> = ({ authenticationService, queryClientProvider }) => {
  const [selectedNotificationPreference, setSelectedNotificationPreference] = useState<NotificationPreferenceType>('emailDigest');
  useLayoutEffect(() => {
    fetchOrganizationBillingSummary(authenticationService.currentSessionToken);
  }, [authenticationService]);
  return <AccountSettingsModalWrapper onDismissCallback={handleModalDismissal} />;
};""",
    ),
    (
        "Turkish (agglutinative)",
        "latin_diacritics",
        "Çekoslovakyalılaştıramadıklarımızdan mısınız? Yönetmelik değişiklikleri doğrultusunda kullanıcı sözleşmelerinin güncellenmesi ve kimlik doğrulama süreçlerinin iyileştirilmesi gerekmektedir.",
    ),
    (
        "C# / .NET Enterprise",
        "camelcase_code",
        """public sealed class OrderFulfillmentBackgroundService : BackgroundService {
    private readonly IServiceScopeFactory _serviceScopeFactory;
    private readonly ILogger<OrderFulfillmentBackgroundService> _logger;
    protected override async Task ExecuteAsync(CancellationToken stoppingToken) {
        using var serviceScope = _serviceScopeFactory.CreateScope();
        var repository = serviceScope.ServiceProvider.GetRequiredService<IOrderFulfillmentRepository>();
        await repository.ProcessPendingShipmentNotificationsAsync(stoppingToken);
    }
}""",
    ),
    (
        "Korean (Hangul)",
        "non_latin_cjk",
        "인공지능과 머신러닝은 문서의 자동 분류, 고객 문의 분석, 이상 거래 탐지 및 실시간 자연어 처리 시스템에서 폭넓게 활용되고 있습니다.",
    ),
    (
        "Czech / Slavic Latin",
        "latin_diacritics",
        "Příliš žlutý člun minul třetí bóji. Nejvyšší správní soud rozhodl o mimořádném opravném prostředku v záležitosti pojišťovnictví a veřejnoprávních smluv mezi obcemi.",
    ),
    (
        "Medical / Pharma English",
        "english_technical",
        "Pharmacokinetics of antihypertensive monotherapy in patients with hypercholesterolemia, glomerulonephritis, electroencephalography abnormalities, and thromboembolism prophylaxis.",
    ),
    (
        "Vietnamese",
        "latin_diacritics",
        "Trí tuệ nhân tạo và học máy được ứng dụng rộng rãi trong phân loại tài liệu tự động, phân tích yêu cầu khách hàng và xử lý ngôn ngữ tự nhiên.",
    ),
    (
        "UUIDs / Hashes / Logs",
        "structured_logs",
        "2026-11-26T16:27:32Z trace_id=a3f1c2e8-550e-41d4-a716-446655440000 sha256=9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08 status=503 latency_ms=1428",
    ),
    (
        "Plain English prose",
        "english_prose",
        "The quick brown fox jumps over the lazy dog. Modern software teams use structured classifiers to route support tickets, flag policy violations, and prioritize urgent customer issues in real time without generating long explanations.",
    ),
    (
        "Python (snake_case)",
        "snake_case_code",
        """def compute_token_usage_summary(request_payload: dict[str, object], max_context_tokens: int = 32736) -> dict[str, int]:
    total_input_tokens = 259
    for question_name, question_spec in request_payload.get("questions", {}).items():
        if not isinstance(question_spec, dict):
            continue
        total_input_tokens += 7 + len(str(question_spec.get("instructions", "")))
    return {"total_input_tokens": total_input_tokens, "remaining_budget": max_context_tokens - total_input_tokens}
""",
    ),
    (
        "Emoji / Social chat",
        "emoji_chat",
        "Breaking news!! 🚀🔥🎉 We just shipped v2.0 😍👏🙌 Check it out 👉💯🥳 Wait what happened 😱🤯😭🙏✨👍",
    ),
]


def query_live(api_key: str, text: str) -> int:
    r = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "typesafe/jev-1.14",
            "state": text,
            "questions": {"q": {"type": "noul", "instructions": "x"}},
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["usage"]["input_tokens"] - 267


def main() -> None:
    jev = JevBillingTokenizer()
    qtok = Tokenizer.from_file(hf_hub_download("Qwen/Qwen3.5-9B", "tokenizer.json"))
    o200k = jev.o200k
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()

    live_counts: dict[str, int] = {}
    if api_key:
        with ThreadPoolExecutor(max_workers=6) as pool:
            for name, cnt in pool.map(lambda w: (w[0], query_live(api_key, w[2])), WORKLOADS):
                live_counts[name] = cnt

    rows = []
    for name, bucket, txt in WORKLOADS:
        q = len(qtok.encode(txt, add_special_tokens=False).ids)
        o = len(o200k.encode(txt, disallowed_special=()))
        pred = jev.count(txt)
        row: dict = {
            "workload": name,
            "bucket": bucket,
            "qwen35_tokens": q,
            "o200k_tokens": o,
            "jev_predicted_tokens": pred,
            "pct_vs_qwen35": round((pred / q - 1) * 100, 1),
            "pct_vs_o200k": round((pred / o - 1) * 100, 1),
            "ratio_vs_qwen35": round(pred / q, 3),
        }
        if api_key:
            row["jev_live_tokens"] = live_counts[name]
            row["live_match"] = live_counts[name] == pred
        rows.append(row)

    rows.sort(key=lambda r: r["ratio_vs_qwen35"], reverse=True)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "live_verified": bool(api_key),
        "live_matched": sum(1 for r in rows if r.get("live_match")) if api_key else None,
        "workloads_total": len(rows),
        "workloads": rows,
    }

    out_path = ROOT / "evidence" / "workload_benchmarks.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{'Workload':30} {'Qwen3.5':>7} {'o200k':>7} {'Jev':>7} {'Live':>6} | {'vs Qwen':>8} {'vs o200k':>9}")
    print("-" * 86)
    for r in rows:
        live_s = str(r["jev_live_tokens"]) if api_key else "-"
        print(
            f"{r['workload']:30} {r['qwen35_tokens']:7d} {r['o200k_tokens']:7d} "
            f"{r['jev_predicted_tokens']:7d} {live_s:>6} | "
            f"{r['pct_vs_qwen35']:+7.1f}% {r['pct_vs_o200k']:+8.1f}%"
        )

    if api_key:
        mismatches = [r for r in rows if not r["live_match"]]
        assert not mismatches, f"Mismatches: {mismatches}"
        print(f"\nALL {len(rows)}/{len(rows)} WORKLOAD BENCHMARKS MATCHED LIVE JEV 1.14 100%.")


if __name__ == "__main__":
    main()
