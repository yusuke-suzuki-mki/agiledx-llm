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
            extracted_text = result.content

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
            chunk_size = 600  # 各チャンクのサイズ
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

    ユーザーの要求：{prompt}

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

    使用例: テキスト
    設定項目:
    type: "string"
    description: 項目の説明を記述
    length: 最大文字数を設定
    unique: 重複不可にする場合にtrue
    default: デフォルト値を設定
    minLengthとmaxLength: 文字数の範囲を設定
    extAttributePhysicalName: パスカルケースの英語名を設定

    使用例: 電話番号
    設定項目:
    type: "string"
    description: 項目の説明を記述
    pattern: 正規表現でフォーマットを指定
    unique: 重複不可にする場合にtrue
    minLengthとmaxLength: 文字数の範囲を設定
    extAttributePhysicalName: パスカルケースの英語名を設定

    使用例: メールアドレス
    設定項目:
    type: "string"
    description: 項目の説明を記述
    format: "email"を指定
    unique: 重複不可にする場合にtrue
    minLengthとmaxLength: 文字数の範囲を設定
    extAttributePhysicalName: パスカルケースの英語名を設定

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

    使用例: フラグ
    設定項目:
    type: "boolean"
    description: 項目の説明を記述
    unique: 重複不可にする場合にtrue
    default: デフォルト値を設定
    extAttributePhysicalName: パスカルケースの英語名を設定

    使用例: バイナリ
    設定項目:
    type: "string"
    description: 項目の説明を記述
    format: "binary"
    mediaType: バイナリデータのメディアタイプを指定
    contentEncoding: "base64"などのエンコーディング方式
    extAttributePhysicalName: パスカルケースの英語名を設定

    使用例: 日時
    設定項目:
    type: "string"
    description: 項目の説明を記述
    format: "date-time"
    default: デフォルト値を設定
    extAttributePhysicalName: パスカルケースの英語名を設定

    使用例: オブジェクト, オブジェクト2
    設定項目:
    type: "object"
    description: 項目の説明を記述
    properties: 子要素を設定
    additionalProperties: 不明なプロパティを許可するか設定
    required: 必須項目をリスト形式で指定（オブジェクト型のみ）
    extAttributePhysicalName: パスカルケースの英語名を設定

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
        messages_for_semantic_answer.append({"role": "system", "content": system_message_for_data_model.format(prompt=prompt)})

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


# --- チャンク前のDocument Intelligenceが抽出したテキストとチャンク後のテキスト情報を出力する関数 ---
@app.function_name('UploadFilesAndCreateChunksZip')
@app.route(route="upload_files_and_create_chunks_zip", auth_level=func.AuthLevel.ANONYMOUS)
def upload_files_and_create_chunks_zip(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a file upload and chunk creation request.')

    try:
        files = req.files.getlist('files')
        if not files:
            return func.HttpResponse("ファイルが見つかりません", status_code=400)

        with tempfile.TemporaryDirectory() as temp_dir:
            zip_filename = os.path.join(temp_dir, 'processed_files.zip')

            with zipfile.ZipFile(zip_filename, 'w') as zipf:
                for file in files:
                    file.stream.seek(0)
                    file_stream = file.stream.read()
                    if not file_stream:
                        return func.HttpResponse("ファイルの読み込みに失敗しました", status_code=400)

                    # Document Intelligenceから取得したテキスト
                    poller = document_intelligence_client.begin_analyze_document(
                        model_id="prebuilt-layout", analyze_request=io.BytesIO(file_stream),
                        content_type="application/octet-stream", output_content_format="markdown"
                    )
                    result = poller.result()

                    if not result or not hasattr(result, 'content'):
                        logging.warning(f"結果が無効、または 'content' 属性が存在しません。ファイル名: {file.filename}")
                        continue

                    extracted_text = result.content if result.content else ""
                    
                    # テーブル情報がある場合のみ処理
                    if hasattr(result, 'tables') and result.tables:
                        for table_idx, table in enumerate(result.tables):
                            markdown_table = []
                            cell_count = len(table.cells)
                            column_count = table.column_count

                            # セルを行ごとに分割
                            index = 0
                            rows = []
                            while index != cell_count:
                                rows.append(
                                    [cell.content for cell in table.cells[index : index + column_count]]
                                )
                                index += column_count

                            # Markdown形式のテーブルを生成
                            markdown_table = ""
                            for i, row in enumerate(rows):
                                for column in row:
                                    markdown_table += f"| {column} "
                                markdown_table += "|\n"
                            
                            # テーブルをMarkdownファイルとして保存
                            table_filename = os.path.join(temp_dir, f"{os.path.splitext(file.filename)[0]}_table_{table_idx + 1}.md")
                            with open(table_filename, 'w', encoding='utf-8') as table_file:
                                table_file.write(markdown_table)
                            zipf.write(table_filename, arcname=f"{os.path.splitext(file.filename)[0]}_table_{table_idx + 1}.md")
                    else:
                        logging.info(f"テーブルが見つかりません。ファイル名: {file.filename}")

                    # チャンキング前のテキストをMarkdownファイルとして保存
                    base_filename = os.path.splitext(file.filename)[0]
                    markdown_filename = os.path.join(temp_dir, f"{base_filename}_extracted.md")
                    with open(markdown_filename, 'w', encoding='utf-8') as markdown_file:
                        markdown_file.write(extracted_text)
                    zipf.write(markdown_filename, arcname=f"{base_filename}_extracted.md")

                    # Markdownチャンキング
                    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
                    md_text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
                    markdown_splits = md_text_splitter.split_text(extracted_text)

                    # チャンク分割
                    chunk_size = 600
                    chunk_overlap = 30
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    splits = text_splitter.split_documents(markdown_splits)

                    # チャンクテキストを結合
                    split_texts = []
                    for j, split in enumerate(splits):
                        split_texts.append(f"--- チャンク {j + 1} ---\n{split.page_content}")
                    combined_text = "\n\n".join(split_texts)

                    # チャンキング後のテキストをテキストファイルに保存
                    chunked_filename = os.path.join(temp_dir, f"{base_filename}_chunked.txt")
                    with open(chunked_filename, 'w', encoding='utf-8') as chunked_file:
                        chunked_file.write(combined_text)
                    zipf.write(chunked_filename, arcname=f"{base_filename}_chunked.txt")

            # ZIPファイルを読み込みレスポンスとして返却
            with open(zip_filename, 'rb') as zip_file:
                zip_data = zip_file.read()

        return func.HttpResponse(
            body=zip_data,
            mimetype="application/zip",
            headers={"Content-Disposition": "attachment; filename=processed_files.zip"},
            status_code=200
        )

    except Exception as e:
        logging.error(f"エラーが発生しました: {e}", exc_info=True)
        return func.HttpResponse(f"エラー: {str(e)}", status_code=500)




# --- Prebuilt-readモデルを使用してチャンク前のDocument Intelligenceが抽出したテキストとチャンク後のテキスト情報を出力する関数 ---
@app.function_name('UploadFilesAndCreateChunksZipUsingReadModel')
@app.route(route="upload_files_and_create_chunks_zip_using_read_model", auth_level=func.AuthLevel.ANONYMOUS)
def upload_files_and_create_chunks_zip(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function started processing a file upload and chunk creation request.')

    try:
        files = req.files.getlist('files')
        if not files:
            logging.warning("No files found in the request.")
            return func.HttpResponse("ファイルが見つかりません", status_code=400)

        logging.info(f"{len(files)} files found in the request.")

        with tempfile.TemporaryDirectory() as temp_dir:
            zip_filename = os.path.join(temp_dir, 'processed_files.zip')
            logging.info(f"Temporary directory created at {temp_dir}")

            with zipfile.ZipFile(zip_filename, 'w') as zipf:
                for file in files:
                    logging.info(f"Processing file: {file.filename}")
                    file.stream.seek(0)
                    file_stream = file.stream.read()
                    if not file_stream:
                        logging.error(f"Failed to read the file stream for file {file.filename}.")
                        return func.HttpResponse(f"ファイル {file.filename} の読み込みに失敗しました", status_code=400)

                    try:
                        poller = document_intelligence_client.begin_analyze_document(
                            model_id="prebuilt-read", analyze_request=io.BytesIO(file_stream),
                            content_type="application/octet-stream", output_content_format="text"
                        )
                        result = poller.result()
                    except Exception as e:
                        logging.error(f"Failed to analyze document {file.filename}: {e}", exc_info=True)
                        continue

                    if not result or 'pages' not in result:
                        logging.warning(f"No valid pages found in the analysis result for file {file.filename}")
                        continue

                    for page in result['pages']:
                        page_number = page.get('pageNumber', 'N/A')
                        words = page.get('words', [])
                        
                        # ページのテキストを生成
                        page_text = ''.join([word['content'] for word in words if word.get('content')])
                        logging.info(f"Generated text for page #{page_number} in file {file.filename}")

                        base_filename = os.path.splitext(file.filename)[0]
                        page_filename = os.path.join(temp_dir, f"{base_filename}_page_{page_number}.txt")
                        try:
                            with open(page_filename, 'w', encoding='utf-8') as page_file:
                                page_file.write(page_text)
                            zipf.write(page_filename, arcname=f"{base_filename}_page_{page_number}.txt")
                            logging.info(f"Saved text for page #{page_number}")
                        except Exception as e:
                            logging.error(f"Failed to save page text for {file.filename} page {page_number}: {e}")
                            continue

                        chunk_size = 500
                        try:
                            chunked_texts = [page_text[i:i + chunk_size] for i in range(0, len(page_text), chunk_size)]
                            chunked_combined_text = "\n\n".join([f"--- チャンク {j + 1} ---\n{chunk}" for j, chunk in enumerate(chunked_texts)])
                        except Exception as e:
                            logging.error(f"Failed to chunk text for {file.filename} page {page_number}: {e}")
                            continue

                        chunked_filename = os.path.join(temp_dir, f"{base_filename}_page_{page_number}_chunked.txt")
                        try:
                            with open(chunked_filename, 'w', encoding='utf-8') as chunked_file:
                                chunked_file.write(chunked_combined_text)
                            zipf.write(chunked_filename, arcname=f"{base_filename}_page_{page_number}_chunked.txt")
                            logging.info(f"Saved chunked text for page #{page_number}")
                        except Exception as e:
                            logging.error(f"Failed to save chunked text for {file.filename} page {page_number}: {e}")
                            continue

            try:
                with open(zip_filename, 'rb') as zip_file:
                    zip_data = zip_file.read()
                logging.info("ZIP file created successfully and ready to be returned.")
            except Exception as e:
                logging.error(f"Failed to read the ZIP file: {e}", exc_info=True)
                return func.HttpResponse("ZIPファイルの読み込みに失敗しました", status_code=500)

        return func.HttpResponse(
            body=zip_data,
            mimetype="application/zip",
            headers={"Content-Disposition": "attachment; filename=processed_files.zip"},
            status_code=200
        )

    except Exception as e:
        logging.error(f"エラーが発生しました: {e}", exc_info=True)
        return func.HttpResponse(f"エラー: {str(e)}", status_code=500)
