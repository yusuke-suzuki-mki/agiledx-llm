import azure.functions as func  # Azure Functions のモジュールをインポート
import logging  # ログ出力を行うためのモジュールをインポート
import os  # 環境変数を操作するためのモジュールをインポート
import io  # バイナリデータを扱うためのモジュール。BytesIOを使用してバイトストリームをメモリ上で操作する。
import json  # JSONデータの操作を行うためのモジュールをインポート
import uuid  # 一意の識別子（UUID）を生成するためのモジュールをインポート
import logging  # ログ出力のために再度インポート（不要なので1つ削除するのが推奨）
import zipfile  # ZIPファイルを操作するためのモジュールをインポート
import tempfile  # 一時ファイルを作成するためのモジュールをインポート
import requests  # HTTPリクエストを送るためのモジュールをインポート
from azure.core.credentials import AzureKeyCredential  # Azureのキー認証を行うためのクラスをインポート
from azure.ai.documentintelligence import DocumentIntelligenceClient  # Azure Form Recognizer クライアントをインポート
from azure.search.documents import SearchClient  # Azure Cognitive Search クライアントをインポート
from azure.search.documents.models import VectorizedQuery  # 検索用のクエリモデルをインポート
from azure.cosmos import CosmosClient  # Azure CosmosDBクライアントをインポート
from openai import AzureOpenAI  # Azure OpenAI のクライアントをインポート
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter  # LangChain のテキスト分割器をインポート


# Azure Functions アプリケーションを定義
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)  # 匿名アクセスを許可するAzure Functionsアプリの設定


# --- 環境変数の読み込み ---
# Azure Cosmos DB のエンドポイント、キー、データベース名、コンテナ名を環境変数から取得
cosmos_db_endpoint = os.environ.get("COSMOSDB_ENDPOINT", "")  # CosmosDBの接続に使用するエンドポイント
cosmos_db_key = os.environ.get("COSMOSDB_KEY", "")  # CosmosDBの認証に使用するキー
cosmos_db_database_name = os.environ.get("COSMOSDB_DATABASE", "")  # 使用するデータベース名
cosmos_db_container_name = os.environ.get("COSMOSDB_CONTAINER", "")  # 使用するコンテナ名

# Azure Storage の接続文字列を環境変数から取得
storage_connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")  # Azure Storageへの接続用

# Azure Form Recognizer のエンドポイントとキーを環境変数から取得
document_intelligence_endpoint = os.environ.get("DOCUMENT_INTELLIGENCE_ENDPOINT", "")  # Form Recognizerのエンドポイント
document_intelligence_key = os.environ.get("DOCUMENT_INTELLIGENCE_KEY", "")  # Form Recognizerの認証キー

# Azure AI Search のエンドポイントとキーを環境変数から取得
search_service_endpoint = os.environ.get("SEARCH_SERVICE_ENDPOINT", "")  # Cognitive Searchのエンドポイント
search_api_key = os.environ.get("SEARCH_API_KEY", "")  # Cognitive Searchの認証キー

# 使用する検索インデックス名を設定
search_index_name = "ragdataset-index-test"  # 検索に使用するインデックスの名前

# Azure OpenAI のエンドポイント、キー、埋め込みモデル名、API バージョンを環境変数から取得
aoai_endpoint = os.environ.get("AOAI_ENDPOINT", "")  # Azure OpenAIのエンドポイント
aoai_api_key = os.environ.get("AOAI_API_KEY", "")  # Azure OpenAIのAPIキー
aoai_embedding_model = os.environ.get("AOAI_EMBEDDING_MODEL_DEPLOYMENT_NAME", "")  # 埋め込みモデルのデプロイ名
aoai_api_version = os.environ.get("AOAI_API_VERSION", "")  # APIのバージョン

# GPT モデルのデプロイ名を環境変数から取得
gpt_deploy = os.environ.get("AOAI_MODEL_DEPLOYMENT_NAME", "")  # 使用するGPTモデルのデプロイ名


# --- クライアントの初期化 ---
# クライアントとは、特定のクラウドサービスに接続し、それを操作するためのインターフェース
# クライアントを介して、データの取得、保存、検索、解析などの操作をプログラムから実行できる

# CosmosDB クライアントの初期化
cosmos_client = CosmosClient(cosmos_db_endpoint, cosmos_db_key)  # CosmosDBへの接続を初期化
# 指定されたデータベースを取得
database = cosmos_client.get_database_client(cosmos_db_database_name)  # CosmosDBのデータベースクライアントを取得
# 指定されたコンテナを取得
container = database.get_container_client(cosmos_db_container_name)  # データ操作用のコンテナクライアントを取得

# Azure OpenAI クライアントの初期化
openai_client = AzureOpenAI(
    api_key=aoai_api_key,  # OpenAIのAPIキーを設定
    azure_endpoint=aoai_endpoint,  # OpenAIのエンドポイントを設定
    api_version=aoai_api_version  # OpenAIのAPIバージョンを設定
)

# Azure Cognitive Search クライアントの初期化
search_client = SearchClient(
    endpoint=search_service_endpoint,  # Cognitive Searchのエンドポイントを設定
    index_name=search_index_name,  # 使用するインデックス名を設定
    credential=AzureKeyCredential(search_api_key)  # 認証に使用するクレデンシャルを設定
)

# Document Intelligence クライアントの初期化
document_intelligence_client = DocumentIntelligenceClient(
    endpoint=document_intelligence_endpoint,  # Form Recognizerのエンドポイントを設定
    credential=AzureKeyCredential(document_intelligence_key)  # 認証に使用するクレデンシャルを設定
)




# 用意されている関数は以下の3つ
# 1. upload_files_and_create_index: ファイルをAzure Storageにアップロードし、インデックスを作成するAPI
# 2. generate_answer: ユーザーの質問に対してRAGを使用して回答を生成するAPI
# 3. process_wikipedia_page: Wikipediaページを取得し、セマンティックチャンキングを行うAPI


# 1 --- アップロードされたファイルをストレージに格納し、並行してAzure AI Searchのインデックスを作成するAPI ---
@app.function_name('UploadFilesAndCreateIndex')  # 関数名を設定
@app.route(route="upload_files_and_create_index", auth_level=func.AuthLevel.ANONYMOUS)  # エンドポイント名とアクセスレベル（匿名アクセス）を設定
def upload_files_and_create_index(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a file upload and index creation request.')  # ログに処理開始を記録

    try:
        # リクエストから複数のファイルを取得
        files = req.files.getlist('files')  # リクエストに含まれるファイルリストを取得
        if not files:  # ファイルが存在しない場合のチェック
            return func.HttpResponse("ファイルが見つかりません", status_code=400)  # ファイルがない場合はエラーレスポンスを返す

        # 各ファイルをAzure Storageにアップロードし、その後インデックスを作成
        for i, file in enumerate(files):  # ファイルを1つずつ処理
            # Azure Blob Storage にファイルをアップロード
            # blob_client = blob_service_client.get_blob_client(container=container_name, blob=file.filename)
            # blob_client.upload_blob(file.stream)
            # logging.info(f"ファイルがAzure Storageにアップロードされました: {file.filename}")

            # Document Intelligence でファイルを処理し、テキストを抽出
            file.stream.seek(0)  # ファイルストリームの先頭に戻す（読み取りのため）
            file_stream = file.stream.read()  # ファイルの内容を読み込む
            if not file_stream:  # ファイルの内容が空の場合
                return func.HttpResponse("ファイルの読み込みに失敗しました", status_code=400)  # エラーレスポンスを返す

            # Form Recognizerでファイルを解析してテキストを抽出
            poller = document_intelligence_client.begin_analyze_document(
                model_id="prebuilt-layout", analyze_request=io.BytesIO(file_stream), content_type="application/octet-stream", output_content_format="markdown"  # プレビルドモデルを使用してレイアウト解析
            )
            result = poller.result()  # 解析結果を取得
            # 抽出されたテキストを格納する変数
            extracted_text = ""

            # ドキュメント内のスタイルを確認
            if result.styles and any([style.is_handwritten for style in result.styles]):
                logging.info("Document contains handwritten content")  # 手書きの内容が含まれていることをログに記録
            else:
                logging.info("Document does not contain handwritten content")  # 手書きの内容が含まれていないことをログに記録

            # 各ページを処理
            for page in result.pages:
                logging.info(f"----Analyzing layout from page #{page.page_number}----")
                logging.info(f"Page has width: {page.width} and height: {page.height}, measured with unit: {page.unit}")

                # ページ内の行ごとのテキストとワード情報を処理
                if page.lines:
                    for line_idx, line in enumerate(page.lines):
                        words = []
                        if page.words:
                            for word in page.words:
                                logging.info(f"......Word '{word.content}' has a confidence of {word.confidence}")
                                if _in_span(word, line.spans):
                                    words.append(word)
                        logging.info(
                            f"...Line # {line_idx} has word count {len(words)} and text '{line.content}' "
                            f"within bounding polygon '{_format_polygon(line.polygon)}'"
                        )
                        # 抽出されたテキストを追加
                        extracted_text += line.content + " "

                # ページ内の選択マーク（チェックボックスなど）の情報を処理
                if page.selection_marks:
                    for selection_mark in page.selection_marks:
                        logging.info(
                            f"Selection mark is '{selection_mark.state}' within bounding polygon "
                            f"'{_format_polygon(selection_mark.polygon)}' and has a confidence of {selection_mark.confidence}"
                        )

            # ドキュメント内の段落情報を処理
            if result.paragraphs:
                logging.info(f"----Detected #{len(result.paragraphs)} paragraphs in the document----")
                # 段落をスパンのオフセット順に並べ替えて順序通りに読み取る
                result.paragraphs.sort(key=lambda p: (p.spans.sort(key=lambda s: s.offset), p.spans[0].offset))
                logging.info("-----Print sorted paragraphs-----")
                for paragraph in result.paragraphs:
                    if not paragraph.bounding_regions:
                        logging.info(f"Found paragraph with role: '{paragraph.role}' within N/A bounding region")
                    else:
                        logging.info(f"Found paragraph with role: '{paragraph.role}' within")
                        logging.info(
                            ", ".join(
                                f" Page #{region.page_number}: {_format_polygon(region.polygon)} bounding region"
                                for region in paragraph.bounding_regions
                            )
                        )
                    logging.info(f"...with content: '{paragraph.content}'")
                    logging.info(f"...with offset: {paragraph.spans[0].offset} and length: {paragraph.spans[0].length}")
                    # 抽出されたテキストを追加
                    extracted_text += paragraph.content + " "

            # ドキュメント内のテーブル情報を処理
            if result.tables:
                for table_idx, table in enumerate(result.tables):
                    logging.info(f"Table # {table_idx} has {table.row_count} rows and {table.column_count} columns")
                    if table.bounding_regions:
                        for region in table.bounding_regions:
                            logging.info(
                                f"Table # {table_idx} location on page: {region.page_number} is {_format_polygon(region.polygon)}"
                            )
                    for cell in table.cells:
                        logging.info(f"...Cell[{cell.row_index}][{cell.column_index}] has text '{cell.content}'")
                        if cell.bounding_regions:
                            for region in cell.bounding_regions:
                                logging.info(
                                    f"...content on page {region.page_number} is within bounding polygon '{_format_polygon(region.polygon)}'"
                                )
                        # 抽出されたテキストを追加
                        extracted_text += cell.content + " "

            # 抽出されたテキストの一部をログに記録
            logging.info(f"抽出されたテキスト: {extracted_text[:500]}...")

            # Markdownベースのセマンティックチャンキング
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]

            logging.info('セマンティックチャンキング開始')  # チャンキング開始をログに記録
            # langchainを活用してセマンティックチャンキングを実施
            md_text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
            markdown_splits = md_text_splitter.split_text(extracted_text)

            # チャンク処理 (RecursiveCharacterTextSplitterを使用)
            chunk_size = 500  # 各チャンクのサイズ
            chunk_overlap = 30  # チャンク間の重複サイズ
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )

            # Markdownで分割されたテキストをさらに小さなチャンクに分割
            splits = text_splitter.split_documents(markdown_splits)
            logging.info(f'セマンティックチャンキング終了{type(splits)}')  # チャンキング終了をログに記録

            # Documentオブジェクトからpage_contentを取り出し、各チャンクをベクトル化してAzure Searchに登録
            for i, split in enumerate(splits):
                chunk = split.page_content  # 各チャンクの内容を取得
                # Azure OpenAIを使用してチャンクをベクトルに変換
                embedding_response = openai_client.embeddings.create(
                    input=chunk,  # チャンク化されたテキストを入力
                    model=aoai_embedding_model  # 使用する埋め込みモデル
                )
                vector = embedding_response.data[0].embedding  # ベクトル化されたデータを取得

                # チャンク化されたテキストとベクトルをAzure Searchに登録
                document = {
                    "id": str(uuid.uuid4()),  # ユニークなIDを付与して一意に識別
                    "content": chunk,  # チャンク化されたテキストを登録
                    "contentVector": vector  # テキストのベクトル化データを登録
                }
                search_client.upload_documents(documents=[document])  # Azure Searchにドキュメントをアップロード
                logging.info(f"チャンクとベクトルがAzure Searchに登録されました: {chunk[:100]}...")  # 登録されたテキストの一部をログに記録

        return func.HttpResponse(f"{len(files)} 個のファイルが正常にアップロードされ、インデックスが作成されました。", status_code=200)  # 成功レスポンスを返す

    except Exception as e:  # 例外が発生した場合の処理
        logging.error(f"エラーが発生しました: {e}")  # エラーメッセージをログに記録
        return func.HttpResponse(f"エラー: {str(e)}", status_code=500)  # エラーレスポンスを返す



# --- ユーザーからの質問文に対してRAGを使用して回答を生成するAPI ---
@app.function_name('GenerateAnswer')  # 関数名を設定
@app.route(route="generate_answer", auth_level=func.AuthLevel.ANONYMOUS)  # エンドポイント名とアクセスレベル（匿名アクセス）を設定
def generate_answer(req: func.HttpRequest) -> func.HttpResponse:

    # AIのキャラクターを決めるためのシステムメッセージを定義する。
    # 検索クエリ作成の専門家
    system_message_for_query = """
    あなたは検索の専門家です。ユーザーの質問内容を分析し、RAG（Retrieval Augmented Generation）検索に最適なクエリを作成してください。クエリは、検索エンジンが関連性の高い情報を正確に引き出せるように設計します。

    以下の指示に従ってクエリを作成してください：
    意図の理解: ユーザーの質問や目的を把握し、必要な情報の種類や検索対象を特定します。具体的な回答や資料を求めている場合は、その内容に沿ったキーワードや概念を抽出してください。
    要素の抽出: 質問の重要なキーワードや概念を特定し、文脈や関連語も加味しながら、簡潔かつ検索に最適化された形で整理します。
    検索精度の最適化: 必要に応じて、ユーザーの質問に関係のない一般的な単語や冗長な表現は省き、情報の関連度を高めるためのキーワードのみを使用します。
    具体性の確保: ユーザーが特定の状況や分野に焦点を当てている場合、その背景に合う単語やフレーズを追加し、情報が絞り込まれるようにしてください。
    出力フォーマット例:
    質問：「新製品の環境への影響について教えてください」
    クエリ：「新製品 環境影響 分析」
    質問内容から必要な要素を抽出し、検索に適したキーワードのみで構成されたクエリを作成してください。
    検索クエリ本文のみ出力してください。「クエリ：」等の部分は必要ありません。本文のみ出力してください
    """

    # NoSQLデータモデルを作成するスペシャリスト
    system_message_for_data_model = """
    あなたは、ユーザーの要求に基づいてNoSQLデータモデルを作成するスペシャリストです。
    ユーザーの要求から最適な情報をRAGから取得してくるので、その取得してきたRAGの情報をもとにまずはシステムの全体のデータモデルを考えてください。
    ただし、考えた結果は100点中60点程度の精度とし、精度を高めるために考えたデータモデルをもとに3回以上データモデルの精査をするようにしてください。
    そして次のルールに従って、最終的なデータモデルのNoSQLデータモデルを設計し、JSON Schema形式で出力してください。JSON Schema形式のデータモデル情報のみ返却してください。

    ルール：

    各エンティティのデータスキーマは、必ずすべての項目を埋めて出力してください。
    階層型データモデルなので、親子関係があったりする場合は1つのエンティティ内にオブジェクトとして子やリストを持つようにしてください。マスタなどは参照（id値をカラムとして持つだけ）してください
    主キー（primaryKey）は、各エンティティで1つだけ設定します。他の項目には設定しないでください。
    出力するJSON Schemaは、以下の要件に従い、適切な制約やデフォルト値を含めてください。
    カラム名は全て日本語で設定し、extAttributePhysicalNameにはパスカルケースで英語名を設定すること。
    DefaultOptionが設定されている場合、そのオプションに応じた追加の項目を設定してください。
    DefaultOptionでAutoincrementを選択した場合は、autoincrement項目も必須です（step_numおよびstart_numを含む）。
    requiredフィールドは、オブジェクト型（object）に対してのみ設定できます。必須フィールドはpropertiesの中にリスト形式で指定してください。配列（array）や他のデータ型には設定しないでください。
    extAttributePhysicalNameは、各フィールドに対応する物理的な属性名をパスカルケースで設定します。各フィールド名の対応関係を明確にしてください。
    displayNameはフィールドの表示名として日本語を設定してください。
    indexやuniqueなど、必要な制約が抜けないように設定してください。
    データ型の使用例と詳細設定
    インテジャー型（number）

    使用例: インテジャー, インテジャー2
    設定項目:
    type: "number"
    description: 項目の説明を記述
    primaryKey: 主キーかどうかを指定（trueまたはfalse）
    unique: 重複不可にする場合にtrue
    defaultOption: "Default"または"Autoincrement"
    autoincrement: 自動インクリメントの設定が必要（step_numとstart_numを指定）
    minimumとmaximum: 値の範囲を設定
    extAttributePhysicalName: パスカルケースの英語名を設定
    テキスト型（string）

    使用例: テキスト
    設定項目:
    type: "string"
    description: 項目の説明を記述
    length: 最大文字数を設定
    unique: 重複不可にする場合にtrue
    default: デフォルト値を設定
    minLengthとmaxLength: 文字数の範囲を設定
    extAttributePhysicalName: パスカルケースの英語名を設定
    電話番号（string, pattern）

    使用例: 電話番号
    設定項目:
    type: "string"
    description: 項目の説明を記述
    pattern: 正規表現でフォーマットを指定
    unique: 重複不可にする場合にtrue
    minLengthとmaxLength: 文字数の範囲を設定
    extAttributePhysicalName: パスカルケースの英語名を設定
    メールアドレス（string, format=email）

    使用例: メールアドレス
    設定項目:
    type: "string"
    description: 項目の説明を記述
    format: "email"を指定
    unique: 重複不可にする場合にtrue
    minLengthとmaxLength: 文字数の範囲を設定
    extAttributePhysicalName: パスカルケースの英語名を設定
    少数（number, decimal）

    使用例: 少数
    設定項目:
    type: "number"
    description: 項目の説明を記述
    mode: "decimal"
    scaleとprecision: 小数点の精度を設定
    unique: 重複不可にする場合にtrue
    default: デフォルト値を設定
    minimumとmaximum: 値の範囲を設定
    extAttributePhysicalName: パスカルケースの英語名を設定
    extNumberFormat: "Decimal"
    価格（number, currency）

    使用例: 価格
    設定項目:
    type: "number"
    description: 項目の説明を記述
    mode: "decimal"
    scaleとprecision: 小数点の精度を設定
    unique: 重複不可にする場合にtrue
    default: デフォルト値を設定
    minimumとmaximum: 値の範囲を設定
    extAttributePhysicalName: パスカルケースの英語名を設定
    extNumberFormat: "Currency"
    フラグ（boolean）

    使用例: フラグ
    設定項目:
    type: "boolean"
    description: 項目の説明を記述
    unique: 重複不可にする場合にtrue
    default: デフォルト値を設定
    extAttributePhysicalName: パスカルケースの英語名を設定
    バイナリ（string, format=binary）

    使用例: バイナリ
    設定項目:
    type: "string"
    description: 項目の説明を記述
    format: "binary"
    mediaType: バイナリデータのメディアタイプを指定
    contentEncoding: "base64"などのエンコーディング方式
    extAttributePhysicalName: パスカルケースの英語名を設定
    日時（string, format=date-time）

    使用例: 日時
    設定項目:
    type: "string"
    description: 項目の説明を記述
    format: "date-time"
    default: デフォルト値を設定
    extAttributePhysicalName: パスカルケースの英語名を設定
    オブジェクト型（object）

    使用例: オブジェクト, オブジェクト2
    設定項目:
    type: "object"
    description: 項目の説明を記述
    properties: 子要素を設定
    additionalProperties: 不明なプロパティを許可するか設定
    required: 必須項目をリスト形式で指定（オブジェクト型のみ）
    extAttributePhysicalName: パスカルケースの英語名を設定
    配列型（array）

    使用例: 配列, 配列2
    設定項目:
    type: "array"
    description: 項目の説明を記述
    items: 配列の要素の型を指定
    extAttributePhysicalName: パスカルケースの英語名を設定
    すべての項目を埋めたJSON Schemaを複数生成し、それぞれ生成されたJson Schemaをlist[]の中に入れてください。Schema毎に"$schema": "http://json-schema.org/draft-04/schema#",を先頭に追加するのを忘れないようにしてください。
    ```json ```という文言は必要ありません。回答はlistの"["から始めて下さい。その他の文言は出力しないでください。
    """

    # ユーザからの質問を元に、Azure AI Searchに投げる検索クエリを生成するためのテンプレートを定義する。
    query_prompt_template = """
    以下のユーザーからの質問に基づいて、検索クエリを生成してください
    例えば、「育児休暇はいつまで取れますか？」という質問があった場合、「育児休暇 取得期間」という形で回答を返してください。

    question: {query}
    """

    try:
        # ---- リクエスト内のプロンプトを取得し、検索用のベクトルデータを作成 ---

        # リクエストボディからユーザープロンプトを取得
        req_body = req.get_json()  # リクエストからJSONデータを取得
        prompt = req_body.get('prompt')  # ユーザーからの質問（プロンプト）を取得
        if not prompt:  # プロンプトが存在しない場合
            return func.HttpResponse("プロンプトが見つかりません。", status_code=400)  # エラーレスポンスを返す

        # セマンティックハイブリッド検索に必要な「ベクトル化されたクエリ」「キーワード検索用クエリ」のうち、ベクトル化されたクエリを生成する。
        response = openai_client.embeddings.create(
            input=prompt,  # ユーザーからの質問を入力としてベクトル化
            model=aoai_embedding_model  # 使用する埋め込みモデル
        )
        # ベクトル化されたクエリを生成し、最も近い5つのコンテンツを検索する設定
        vector_query = VectorizedQuery(vector=response.data[0].embedding, k_nearest_neighbors=5, fields="contentVector")

        # ユーザーからの質問を元に、Azure AI Searchに投げる検索クエリを生成する。
        messages_for_search_query = query_prompt_template.format(query=prompt)  # プロンプトテンプレートに質問を埋め込む


        # ---- Azure AI Search内の情報を検索するためにGPTを使用して検索用の文章を洗練させる ----

        # Azure OpenAI に検索用クエリ生成を依頼
        response = openai_client.chat.completions.create(
            model=gpt_deploy,  # 使用するGPTモデル
            messages=[
                {"role": "system", "content": system_message_for_query},  # システムメッセージを設定
                {"role": "user", "content": messages_for_search_query}  # ユーザーの質問を設定
            ]
        )

        # 生成された検索クエリを取得
        search_query = response.choices[0].message.content.strip()  # クエリのテキストを取得
        logging.info(f"生成された検索クエリ: {search_query}")  # ログにクエリを記録


        # ---- Azure AI Searchに対してセマンティックハイブリッド検索を行い、データモデル作成に必要な情報を取得する ----

        # 「ベクトル化されたクエリ」「キーワード検索用クエリ」を用いて、Azure AI Searchに対してセマンティックハイブリッド検索を行う。
        results = search_client.search(
            query_type='semantic',  # セマンティック検索を使用
            semantic_configuration_name='ragdataset-semantic',  # 使用するセマンティック設定名
            search_text=search_query,  # 生成された検索クエリ
            vector_queries=[vector_query],  # ベクトル化されたクエリを指定
            select=['id', 'content'],  # 取得するフィールドを指定
            query_caption='extractive',  # 抽出キャプションを使用
            query_answer="extractive",  # 抽出回答を使用
            highlight_pre_tag='<em>',  # ハイライトの開始タグ
            highlight_post_tag='</em>',  # ハイライトの終了タグ
            top=100  # 上位100個の結果を取得
        )

        # セマンティックアンサーを取得する
        semantic_answers = results.get_answers()  # 検索結果からセマンティックアンサーを取得


        # ---- Azure AI Searchの検索結果を用いてGPTにデータモデル生成を依頼する ----

        # 回答生成用のメッセージリスト
        messages_for_semantic_answer = []

        # システムメッセージを追加（生成する回答の指示）
        messages_for_semantic_answer.append({"role": "system", "content": system_message_for_data_model})

        # セマンティックアンサーの有無で返答を変える
        user_message = ""

        # semantic_answersがNoneでないか、リストが空でないか確認
        if semantic_answers is None or len(semantic_answers) == 0:
            # セマンティックアンサーがない場合の処理
            sources = ["[Source " + result["id"] + "]: " + result["content"] for result in results]  # 検索結果をソースリストに追加
            source = "\n".join(sources)  # ソースリストを結合

            # ユーザーメッセージに検索結果のソースを埋め込む
            user_message = """
            {query}

            Sources:
            {source}
            """.format(query=search_query, source=source)  # 検索結果のクエリとソースを埋め込んだメッセージ
        else:
            # セマンティックアンサーがある場合、その内容を使用
            user_message = """
            {query}

            Sources:
            {source}
            """.format(query=search_query, source=semantic_answers[0].text)  # 検索結果のクエリと最初のアンサーを埋め込んだメッセージ

        # ユーザーからの入力メッセージとして設定
        messages_for_semantic_answer.append({"role": "user", "content": user_message})

        logging.info(f"最終的にGPTに渡されるプロンプト: {messages_for_semantic_answer}")  # ログにクエリを記録

        # Azure OpenAIを使って最終的な回答を生成
        response = openai_client.chat.completions.create(
            model=gpt_deploy,  # 使用するGPTモデル
            messages=messages_for_semantic_answer  # 生成されたメッセージリストを入力
        )

        # 生成された回答を取得
        generated_answer = response.choices[0].message.content.strip()  # 回答テキストを取得
        logging.info(f"生成された回答: {generated_answer}")  # ログに回答を記録


        # ---- GPTの回答結果からJson Schemaを抽出し、Json Schemaごとにファイルを作成、zip形式で圧縮する ----

        # JSONが空でないかチェック
        if not generated_answer:  # 回答が空の場合
            logging.error("生成された回答が空です")  # エラーメッセージをログに記録
            return func.HttpResponse(
                json.dumps({"error": "生成された回答が空です"}, ensure_ascii=False),  # エラーメッセージを返す
                mimetype="application/json",
                status_code=400
            )

        # 生成された回答をJSONとしてパース
        try:
            json_schemas = json.loads(generated_answer)  # 生成された回答をJSONとして読み込む
        except json.JSONDecodeError as e:  # JSON形式のエラーをキャッチ
            logging.error(f"JSONのパースに失敗しました: {str(e)}")  # エラーメッセージをログに記録
            return func.HttpResponse(
                json.dumps({"error": "無効なJSON形式です"}, ensure_ascii=False),  # エラーメッセージを返す
                mimetype="application/json",
                status_code=400
            )

        # 一時ディレクトリを作成
        with tempfile.TemporaryDirectory() as temp_dir:  # 一時フォルダのコンテキストマネージャーを使用
            # ZIPファイルの保存先
            zip_filename = os.path.join(temp_dir, 'schemas.zip')  # ZIPファイルのパスを設定

            # プロンプトをテキストファイルとして保存
            prompt_filename = os.path.join(temp_dir, 'final_prompt.txt')  # プロンプトファイル名を設定
            with open(prompt_filename, 'w', encoding='utf-8') as prompt_file:
                # messages_for_semantic_answer の内容をテキストファイルに書き込む
                for message in messages_for_semantic_answer:
                    prompt_file.write(f"{message['role']}: {message['content']}\n")

            # ZIPファイルを作成してスキーマファイルとプロンプトファイルを追加
            with zipfile.ZipFile(zip_filename, 'w') as zipf:  # ZIPファイルを作成
                # プロンプトファイルをZIPに追加
                zipf.write(prompt_filename, arcname='final_prompt.txt')  # テキストファイルをZIPに追加

                for index, schema in enumerate(json_schemas):  # 各スキーマを処理
                    # スキーマファイル名を生成
                    schema_filename = f"schema_{index + 1}.json"  # ファイル名を生成
                    schema_path = os.path.join(temp_dir, schema_filename)  # フルパスを生成

                    # 個別のスキーマファイルを保存
                    with open(schema_path, 'w', encoding='utf-8') as f:  # JSONファイルを作成
                        json.dump(schema, f, ensure_ascii=False, indent=4)  # JSONデータを書き込む

                    # ZIPファイルにスキーマファイルを追加
                    zipf.write(schema_path, arcname=schema_filename)  # ZIPに追加

            # ZIPファイルを読み込みレスポンスとして返却
            with open(zip_filename, 'rb') as zip_file:  # ZIPファイルをバイナリで読み込み
                zip_data = zip_file.read()  # ZIPファイルの内容を読み込む

        # ---- 作成したzipファイルをレスポンスとして返却 ----

        # HTTPレスポンスを返却
        return func.HttpResponse(
            zip_data,  # ZIPファイルのデータを返す
            mimetype="application/zip",  # MIMEタイプをZIPに設定
            headers={"Content-Disposition": "attachment; filename=schemas.zip"},  # ファイルのダウンロード名を設定
            status_code=200  # 成功ステータスコード
        )

    except Exception as e:  # エラー発生時の処理
        logging.error(f"エラーが発生しました: {str(e)}")  # エラーメッセージをログに記録
        return func.HttpResponse(
            json.dumps({"error": "エラーが発生しました"}, ensure_ascii=False),  # エラーメッセージをJSONで返す
            mimetype="application/json",
            status_code=500  # サーバーエラーステータスコード
        )


# --- セマンティックハイブリッド検索の結果を返却する関数 ---
@app.function_name('TestGenerateSearchResults')  # 関数名を設定
@app.route(route="test_generate_search_results", auth_level=func.AuthLevel.ANONYMOUS)  # エンドポイント名とアクセスレベル（匿名アクセス）を設定
def test_generate_search_results(req: func.HttpRequest) -> func.HttpResponse:

    # AIのキャラクターを決めるためのシステムメッセージを定義する。
    # ここでは検索クエリ作成の専門家として動作させる想定
    system_message_for_query = """
    あなたは検索の専門家です。ユーザーの質問内容を分析し、RAG（Retrieval Augmented Generation）検索に最適なクエリを作成してください。クエリは、検索エンジンが関連性の高い情報を正確に引き出せるように設計します。

    以下の指示に従ってクエリを作成してください：
    意図の理解: ユーザーの質問や目的を把握し、必要な情報の種類や検索対象を特定します。具体的な回答や資料を求めている場合は、その内容に沿ったキーワードや概念を抽出してください。
    要素の抽出: 質問の重要なキーワードや概念を特定し、文脈や関連語も加味しながら、簡潔かつ検索に最適化された形で整理します。
    検索精度の最適化: 必要に応じて、ユーザーの質問に関係のない一般的な単語や冗長な表現は省き、情報の関連度を高めるためのキーワードのみを使用します。
    具体性の確保: ユーザーが特定の状況や分野に焦点を当てている場合、その背景に合う単語やフレーズを追加し、情報が絞り込まれるようにしてください。
    出力フォーマット例:
    質問：「新製品の環境への影響について教えてください」
    クエリ：「新製品 環境影響 分析」
    質問内容から必要な要素を抽出し、検索に適したキーワードのみで構成されたクエリを作成してください。
    検索クエリ本文のみ出力してください。「クエリ：」等の部分は必要ありません。本文のみ出力してください
    """
    try:
        # リクエストボディからユーザープロンプトを取得
        req_body = req.get_json()
        prompt = req_body.get('prompt')
        if not prompt:
            return func.HttpResponse("プロンプトが見つかりません。", status_code=400)

        # セマンティックハイブリッド検索に必要な「ベクトル化されたクエリ」「キーワード検索用クエリ」のうち、ベクトル化されたクエリを生成する。
        response = openai_client.embeddings.create(
            input=prompt,  # ユーザーからの質問を入力としてベクトル化
            model=aoai_embedding_model  # 使用する埋め込みモデル
        )
        # ベクトル化されたクエリを生成し、最も近い5つのコンテンツを検索する設定
        vector_query = VectorizedQuery(vector=response.data[0].embedding, k_nearest_neighbors=5, fields="contentVector")

        # ユーザーからの質問を元に、Azure AI Searchに投げる検索クエリを生成する
        query_prompt_template = "以下のユーザーからの質問に基づいて、検索クエリを生成してください: {query}"
        messages_for_search_query = query_prompt_template.format(query=prompt)  # プロンプトテンプレートに質問を埋め込む


        # ---- Azure AI Search内の情報を検索するためにGPTを使用して検索用の文章を洗練させる ----

        # Azure OpenAI に検索用クエリ生成を依頼
        response = openai_client.chat.completions.create(
            model=gpt_deploy,  # 使用するGPTモデル
            messages=[
                {"role": "system", "content": system_message_for_query},  # システムメッセージを設定
                {"role": "user", "content": messages_for_search_query}  # ユーザーの質問を設定
            ]
        )

        # 生成された検索クエリを取得
        search_query = response.choices[0].message.content.strip()  # クエリのテキストを取得

        # Azure AI Searchに対してセマンティックハイブリッド検索を実行
        results = search_client.search(
            query_type='semantic',
            semantic_configuration_name='ragdataset-semantic',
            search_text=search_query,
            vector_queries=[vector_query],
            select=['id', 'content'],
            query_caption='extractive',
            query_answer="extractive",
            highlight_pre_tag='<em>',
            highlight_post_tag='</em>',
            top=100  # 取得する結果数を設定
        )

        # 検索結果をすべて取得してリストにまとめる
        all_results = [{"id": result["id"], "content": result["content"]} for result in results]

        # クエリと検索結果を1つのテキストファイルに書き込む
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
            temp_file.write(f"検索クエリ:\n{search_query}\n\n".encode('utf-8'))
            temp_file.write("検索結果:\n".encode('utf-8'))
            for result in all_results:
                temp_file.write(f"ID: {result['id']}\n内容: {result['content']}\n\n".encode('utf-8'))
            temp_file_path = temp_file.name  # テキストファイルのパスを取得

        # テキストファイルを読み込みレスポンスとして返却
        with open(temp_file_path, 'rb') as file:
            file_data = file.read()

        # ファイル削除
        os.remove(temp_file_path)

        # HTTPレスポンスを返却
        return func.HttpResponse(
            file_data,  # テキストファイルのデータを返す
            mimetype="text/plain",  # MIMEタイプをテキストに設定
            headers={"Content-Disposition": "attachment; filename=search_results.txt"},  # ファイルのダウンロード名を設定
            status_code=200  # 成功ステータスコード
        )

    except Exception as e:
        logging.error(f"エラーが発生しました: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": "エラーが発生しました"}, ensure_ascii=False),
            mimetype="application/json",
            status_code=500
        )



# --- セマンティックチャンキングの実装例 ---
@app.route(route="process_wikipedia_page", methods=['GET'], auth_level=func.AuthLevel.ANONYMOUS)  # エンドポイント名とアクセスレベル（匿名アクセス）を設定
def process_wikipedia_page(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Azure Function for processing Wikipedia page and indexing called.')  # ログに処理開始を記録

    def get_wikipedia_page(title: str):
        """
        Retrieve the full text content of a Wikipedia page.

        :param title: str - Title of the Wikipedia page.
        :return: str - Full text content of the page as raw string.
        """
        # Wikipedia APIのエンドポイント
        URL = "https://ja.wikipedia.org/w/api.php"

        # APIリクエストのパラメータを設定
        params = {
            "action": "query",  # クエリアクション
            "format": "json",  # レスポンスフォーマット
            "titles": title,  # 取得するページのタイトル
            "prop": "extracts",  # 抽出したテキストを取得
            "explaintext": True,  # HTMLタグを除去したプレーンテキスト
        }

        # Wikipediaのベストプラクティスに従ってUser-Agentヘッダーを設定
        headers = {"User-Agent": "tutorial/0.0.1"}

        response = requests.get(URL, params=params, headers=headers)  # APIリクエストを送信
        data = response.json()  # レスポンスをJSON形式で取得

        # ページの内容を抽出
        page = next(iter(data["query"]["pages"].values()))  # 取得したページ情報を取得
        return page["extract"] if "extract" in page else None  # テキストが存在すれば返す

    try:
        # GETリクエストからtitleを取得
        title = req.params.get('title', '葬送のフリーレン')  # 'title' パラメータを取得、デフォルトは"葬送のフリーレン"
        logging.info(f"Fetching Wikipedia page for title: {title}")  # ログにリクエストされたタイトルを記録
        
        # Wikipediaから指定されたページの内容を取得
        full_document = get_wikipedia_page(title)  # ページ内容を取得

        if not full_document:  # ページが存在しない場合のチェック
            return func.HttpResponse(f"Wikipedia page '{title}' not found", status_code=404)  # 404エラーレスポンスを返す

        # Markdownベースのセマンティックチャンキング
        headers_to_split_on = [
            ("#", "Header 1"),  # ヘッダーレベル1（#）
            ("##", "Header 2"),  # ヘッダーレベル2（##）
            ("###", "Header 3"),  # ヘッダーレベル3（###）
        ]

        logging.info('セマンティックチャンキング開始')  # チャンキング開始をログに記録
        # langchainを活用してセマンティックチャンキングを実施
        md_text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)  # ヘッダーごとに分割する設定
        markdown_splits = md_text_splitter.split_text(full_document)  # Markdown形式のテキストを分割

        # チャンク処理 (RecursiveCharacterTextSplitterを使用)
        chunk_size = 500  # 各チャンクのサイズ
        chunk_overlap = 30  # チャンク間の重複サイズ
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap  # チャンクサイズとオーバーラップサイズを指定
        )

        # Markdownで分割されたテキストをさらに小さなチャンクに分割
        splits = text_splitter.split_documents(markdown_splits)  # チャンクごとに分割
        logging.info(f'セマンティックチャンキング終了{type(splits)}')  # チャンキング終了をログに記録

        # Documentオブジェクトからpage_contentを取り出し、区切りを入れて結合
        split_texts = []  # 分割されたテキストを格納するリスト
        for i, split in enumerate(splits):  # 各チャンクを処理
            split_texts.append(f"--- チャンク {i+1} ---\n{split.page_content}")  # チャンク番号と内容を追加

        # チャンク化されたテキストを結合
        combined_text = "\n\n".join(split_texts)  # チャンクを結合して1つの文字列にする

        # チャンク化されたテキストを返却
        return func.HttpResponse(
            body=combined_text,  # チャンク化されたテキストをレスポンスの本文に設定
            mimetype="text/plain",  # MIMEタイプをプレーンテキストに設定
            status_code=200  # 成功ステータスコードを返す
        )

    except Exception as e:  # エラー発生時の処理
        logging.error(f"エラーが発生しました: {e}")  # エラーメッセージをログに記録
        return func.HttpResponse(f"エラー: {str(e)}", status_code=500)  # 500エラーレスポンスを返す
