import azure.functions as func
import logging
import os
import json
import uuid
import logging
import tempfile
import requests
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient  # Azure Form Recognizer
from azure.search.documents import SearchClient  # Azure Cognitive Search
from azure.search.documents.models import VectorizedQuery
from azure.cosmos import CosmosClient  # CosmosDBクライアント
from openai import AzureOpenAI  # Azure OpenAIのクライアント
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from azure.ai.documentintelligence import DocumentIntelligenceClient


app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# CosmosDB の接続情報を環境変数から取得
cosmos_db_endpoint = os.environ.get("COSMOSDB_ENDPOINT", "")
cosmos_db_key = os.environ.get("COSMOSDB_KEY", "")
cosmos_db_database_name = os.environ.get("COSMOSDB_DATABASE", "")
cosmos_db_container_name = os.environ.get("COSMOSDB_CONTAINER", "")

# CosmosDB クライアントを作成
cosmos_client = CosmosClient(cosmos_db_endpoint, cosmos_db_key)
database = cosmos_client.get_database_client(cosmos_db_database_name)
container = database.get_container_client(cosmos_db_container_name)

# 環境変数から各種設定を取得
storage_connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
form_recognizer_endpoint = os.environ.get("FORM_RECOGNIZER_ENDPOINT", "")
form_recognizer_key = os.environ.get("FORM_RECOGNIZER_KEY", "")
search_service_endpoint = os.environ.get("SEARCH_SERVICE_ENDPOINT", "")
search_api_key = os.environ.get("SEARCH_API_KEY", "")
search_index_name = "ragdataset-index-test"  # 固定値として定義
aoai_endpoint = os.environ.get("AOAI_ENDPOINT", "")
aoai_api_key = os.environ.get("AOAI_API_KEY", "")
aoai_embedding_model = os.environ.get("AOAI_EMBEDDING_MODEL_DEPLOYMENT_NAME", "")
aoai_api_version = os.environ.get("AOAI_API_VERSION", "")
gpt_deploy = os.environ.get("AOAI_MODEL_DEPLOYMENT_NAME", "")

# Azure OpenAI クライアント
openai_client = AzureOpenAI(
    api_key=aoai_api_key,
    azure_endpoint=aoai_endpoint,
    api_version=aoai_api_version
)

# Azure Search クライアント
search_client = SearchClient(
    endpoint=search_service_endpoint,
    index_name=search_index_name,
    credential=AzureKeyCredential(search_api_key)
)

# Document Intelligence クライアント
form_recognizer_client = DocumentAnalysisClient(
    endpoint=form_recognizer_endpoint,
    credential=AzureKeyCredential(form_recognizer_key)
)



# --- アップロードされたファイルをストレージに格納し、並行してAzure AI Searchのインデックスを作成するAPI ---
@app.function_name('UploadFilesAndCreateIndex')
@app.route(route="upload_files_and_create_index", auth_level=func.AuthLevel.ANONYMOUS)
def upload_files_and_create_index(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a file upload and index creation request.')

    # --- 補助関数 ---
    def chunk_text(text: str, chunk_size: int = 500) -> list:
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        return chunks

    try:
        # リクエストから複数のファイルを取得
        files = req.files.getlist('files')
        if not files:
            return func.HttpResponse("ファイルが見つかりません", status_code=400)

        # 各ファイルをAzure Storageにアップロードし、その後インデックスを作成
        for i, file in enumerate(files):
            # Azure Blob Storage にファイルをアップロード
            # blob_client = blob_service_client.get_blob_client(container=container_name, blob=file.filename)
            # blob_client.upload_blob(file.stream)
            # logging.info(f"ファイルがAzure Storageにアップロードされました: {file.filename}")

            # Document Intelligence でファイルを処理し、テキストを抽出
            # ファイルストリームを一度リセット（先頭に戻す）
            file.stream.seek(0)
            file_stream = file.stream.read()
            if not file_stream:
                return func.HttpResponse("ファイルの読み込みに失敗しました", status_code=400)

            poller = form_recognizer_client.begin_analyze_document(
                model_id="prebuilt-layout", document=file_stream, output_content_format="markdown"
            )
            result = poller.result()
            extracted_text = ""
            for page in result.pages:
                for line in page.lines:
                    extracted_text += line.content + " "  # 各行のテキストを結合していく
            logging.info(f"抽出されたテキスト: {extracted_text[:500]}...")

            # 抽出されたテキストをチャンク化してベクトル化
            text_chunks = chunk_text(extracted_text)
            for chunk in text_chunks:
                embedding_response = openai_client.embeddings.create(
                    input=chunk,
                    model=aoai_embedding_model
                )
                vector = embedding_response.data[0].embedding

                # チャンク化されたテキストとベクトルをAzure Searchに登録
                document = {
                    "id": str(uuid.uuid4()),  # ユニークなIDを付与
                    "content": chunk,
                    "contentVector": vector
                }
                search_client.upload_documents(documents=[document])
                logging.info(f"チャンクとベクトルがAzure Searchに登録されました: {chunk[:100]}...")

        return func.HttpResponse(f"{len(files)} 個のファイルが正常にアップロードされ、インデックスが作成されました。", status_code=200)

    except Exception as e:
        logging.error(f"エラーが発生しました: {e}")
        return func.HttpResponse(f"エラー: {str(e)}", status_code=500)



# # --- ragdataset内に存在するドキュメントをすべて取得し、Azure AI Searchのインデックスを再作成する処理 ---
# @app.function_name('RecreateSearchIndex')
# @app.route(route="recreate_search_index", auth_level=func.AuthLevel.ANONYMOUS)
# def recreate_search_index(req: func.HttpRequest) -> func.HttpResponse:

#     # --- 補助関数 ---
#     def chunk_text(text: str, chunk_size: int = 500) -> list:
#         chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
#         return chunks
    
#     try:
#         # Blob コンテナ内のすべてのファイルを取得
#         blobs_list = blob_service_client.get_container_client(container_name).list_blobs()

#         for blob in blobs_list:
#             # 各ファイルの内容をダウンロード
#             blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob.name)
#             blob_content = blob_client.download_blob().readall()

#             # Document Intelligence でテキストを抽出
#             poller = form_recognizer_client.begin_analyze_document(
#                 model_id="prebuilt-document", document=blob_content
#             )
#             result = poller.result()
#             extracted_text = " ".join([page.content for page in result.pages])

#             # チャンク化してベクトル化
#             text_chunks = chunk_text(extracted_text)
#             vectorized_chunks = []
#             for chunk in text_chunks:
#                 embedding_response = openai_client.embeddings.create(
#                     input=chunk,
#                     model=aoai_embedding_model
#                 )
#                 vectorized_chunks.append(embedding_response.data[0].embedding)

#             # インデックスに登録
#             for chunk, vector in zip(text_chunks, vectorized_chunks):
#                 search_document = {
#                     "content": chunk,
#                     "contentVector": vector
#                 }
#                 search_client.upload_documents(documents=[search_document])

#         return func.HttpResponse("インデックスの作成が完了しました。", status_code=200)

#     except Exception as e:
#         logging.error(f"エラーが発生しました: {e}")
#         return func.HttpResponse(f"エラー: {str(e)}", status_code=500)


# # --- ユーザーからの質問文に対してRAGを使用して回答を生成するAPI ---
@app.function_name('GenerateAnswer')
@app.route(route="generate_answer", auth_level=func.AuthLevel.ANONYMOUS)
def generate_answer(req: func.HttpRequest) -> func.HttpResponse:

    # AIのキャラクターを決めるためのシステムメッセージを定義する。
    system_message = """
    あなたは、ユーザーの要求に基づいてNoSQLデータモデルを作成するスペシャリストです。次のルールに従って、NoSQLデータモデルを設計し、JSON Schema形式で出力してください。

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
    すべての項目を埋めたJSON Schemaを生成し、それぞれのファイルを作成してzip形式でダウンロードできるようにしてください。
    """

    # ユーザからの質問を元に、Azure AI Searchに投げる検索クエリを生成するためのテンプレートを定義する。
    query_prompt_template = """
    以下のユーザーからの質問に基づいて、検索クエリを生成してください
    例えば、「育児休暇はいつまで取れますか？」という質問があった場合、「育児休暇 取得期間」という形で回答を返してください。

    question: {query}
    """

    def save_to_cosmos(user_input: str, generated_answer: str):
        item = {'id': str(uuid.uuid4()), 'user_input': user_input, 'generated_answer': generated_answer}
        container.create_item(body=item)

    try:
        # リクエストボディからユーザープロンプトを取得
        req_body = req.get_json()
        prompt = req_body.get('prompt')
        if not prompt:
            return func.HttpResponse("プロンプトが見つかりません。", status_code=400)

        # セマンティックハイブリッド検索に必要な「ベクトル化されたクエリ」「キーワード検索用クエリ」のうち、ベクトル化されたクエリを生成する。
        response = openai_client.embeddings.create(
            input = prompt,
            model = aoai_embedding_model
        )
        vector_query = VectorizedQuery(vector=response.data[0].embedding, k_nearest_neighbors=3, fields="contentVector")

        # ユーザーからの質問を元に、Azure AI Searchに投げる検索クエリを生成する。
        # セマンティックハイブリッド検索に必要な「ベクトル化されたクエリ」「キーワード検索用クエリ」のうち、検索クエリを生成する。
        messages_for_search_query = query_prompt_template.format(query=prompt)

        # Azure OpenAI に検索用クエリ生成を依頼
        response = openai_client.chat.completions.create(
            model=gpt_deploy,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": messages_for_search_query}
            ]
        )

        # 生成された検索クエリを取得
        search_query = response.choices[0].message.content.strip()
        logging.info(f"生成された検索クエリ: {search_query}")

        # 「ベクトル化されたクエリ」「キーワード検索用クエリ」を用いて、Azure AI Searchに対してセマンティックハイブリッド検索を行う。
        # Azure AI Searchにセマンティックハイブリッド検索を行う
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
            top=2
        )

        # セマンティックアンサーを取得する
        semantic_answers = results.get_answers()

        # 回答生成用のメッセージリスト
        messages_for_semantic_answer = []

        # システムメッセージを追加（生成する回答の指示）
        messages_for_semantic_answer.append({"role": "system", "content": system_message})

        # セマンティックアンサーの有無で返答を変える
        user_message = ""

        # semantic_answersがNoneでないか、リストが空でないか確認
        if semantic_answers is None or len(semantic_answers) == 0:
            # セマンティックアンサーがない場合の処理
            sources = ["[Source " + result["id"] + "]: " + result["content"] for result in results]
            source = "\n".join(sources)

            # ユーザーメッセージに検索結果のソースを埋め込む
            user_message = """
            {query}

            Sources:
            {source}
            """.format(query=search_query, source=source)
        else:
            # セマンティックアンサーがある場合、その内容を使用
            user_message = """
            {query}

            Sources:
            {source}
            """.format(query=search_query, source=semantic_answers[0].text)

        # ユーザーからの入力メッセージとして設定
        messages_for_semantic_answer.append({"role": "user", "content": user_message})

        # Azure OpenAIを使って最終的な回答を生成
        response = openai_client.chat.completions.create(
            model=gpt_deploy,
            messages=messages_for_semantic_answer
        )

        # 生成された回答を取得
        generated_answer = response.choices[0].message.content.strip()
        logging.info(f"生成された回答: {generated_answer}")

        # プロンプトと生成された回答を CosmosDB に保存
        save_to_cosmos(prompt, generated_answer)

        return func.HttpResponse(
            json.dumps({"answer": generated_answer}, ensure_ascii=False),
            mimetype="application/json; charset=utf-8",
            status_code=200
        )

    except Exception as e:
        logging.error(f"エラーが発生しました: {e}")
        return func.HttpResponse(f"エラー: {str(e)}", status_code=500)






# 以下は動作確認用の関数


# --- Blob クライアントの接続を確認 ---
# @app.route(route="verify_blob_connection", auth_level=func.AuthLevel.ANONYMOUS)
# def verify_blob_connection(req: func.HttpRequest) -> func.HttpResponse:
#     """
#     Azure Blob Storageへの接続を確認し、ファイルをアップロードするためのAzure Functionです。
#     """
#     logging.info('Azure Blob Storageへの接続確認リクエストを受信しました。')

#     try:
#         # リクエストから複数のファイルを取得
#         files = req.files.getlist('files')
#         if not files:
#             return func.HttpResponse("ファイルが見つかりません", status_code=400)

#         # ファイルアップロードの結果を保持するリスト
#         upload_results = []

#         # 各ファイルをAzure Storageにアップロードし、その後インデックスを作成
#         for file in files:
#             try:
#                 # Azure Blob Storage にファイルをアップロード
#                 blob_client = blob_service_client.get_blob_client(container=container_name, blob=file.filename)
#                 blob_client.upload_blob(file.stream)
#                 logging.info(f"ファイルがAzure Storageにアップロードされました: {file.filename}")
#                 upload_results.append(f"アップロード成功: {file.filename}")
#             except Exception as upload_error:
#                 logging.error(f"ファイルのアップロードに失敗しました: {file.filename}, エラー: {upload_error}")
#                 upload_results.append(f"アップロード失敗: {file.filename}, エラー: {upload_error}")

#         # 処理結果をまとめて返却
#         result_message = "\n".join(upload_results)
#         return func.HttpResponse(result_message, status_code=200)

#     except Exception as e:
#         logging.error(f"Blob クライアントへの接続に失敗しました: {e}")
#         return func.HttpResponse(f"Blob クライアントへの接続に失敗しました: {str(e)}", status_code=500)



# # GPTの埋め込みモデルの動作確認
# @app.route(route="upload_and_embed")
# def upload_and_embed(req: func.HttpRequest) -> func.HttpResponse:
#     logging.info('ファイルの埋め込み生成を開始します。')

#     try:
#         # リクエストからファイルを取得
#         file = req.files.get('file')
#         if not file:
#             return func.HttpResponse(
#                 "ファイルが見つかりません。リクエストに 'file' を含めてください。",
#                 status_code=400
#             )

#         # Document Intelligence でファイルを処理し、テキストを抽出
#         file_stream = file.stream.read()
#         poller = form_recognizer_client.begin_analyze_document(
#             model_id="prebuilt-document", document=file_stream
#         )
#         result = poller.result()

#         # ページごとの行を結合してテキストを抽出
#         extracted_text = ""
#         for page in result.pages:
#             for line in page.lines:
#                 extracted_text += line.content + " "
#         logging.info(f"抽出されたテキスト: {extracted_text[:500]}...")

#         # 抽出されたテキストをチャンク化してベクトル化
#         text_chunks = chunk_text(extracted_text)
#         vectorized_chunks = []
#         for chunk in text_chunks:
#             embedding_response = openai_client.embeddings.create(
#                 input=chunk,
#                 model=aoai_embedding_model
#             )
#             vectorized_chunks.append(embedding_response.data[0].embedding)

#         # response.data から埋め込みベクトルを取得
#             embedding_vector = embedding_response.data[0].embedding
#             vectorized_chunks.append(embedding_vector)

#         logging.info(f"生成された埋め込みベクトル: {vectorized_chunks[:1]}...")

#         # ベクトルデータをレスポンスとして返却
#         return func.HttpResponse(f"埋め込みベクトル: {vectorized_chunks}", status_code=200)

#     except Exception as e:
#         logging.error(f"エラーが発生しました: {e}")
#         return func.HttpResponse(f"エラー: {str(e)}", status_code=500)
    



# BATCH_SIZE = 3  # 一度にアップロードするファイルの数

# # --- 非同期バッチアップロード関数 ---
# @app.route(route="upload_files_async_batch", auth_level=func.AuthLevel.ANONYMOUS)
# async def upload_files_async_batch(req: func.HttpRequest) -> func.HttpResponse:
#     logging.info('Python HTTP trigger function processed a file upload request with async batch processing.')

#     try:
#         # リクエストから複数のファイルを取得
#         files = req.files.getlist('files')
#         if not files:
#             return func.HttpResponse("ファイルが見つかりません", status_code=400)

#         # Blob Service Client を非同期で取得
#         blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)

#         # バッチ処理の非同期アップロードタスクを作成
#         for i in range(0, len(files), BATCH_SIZE):
#             batch = files[i:i + BATCH_SIZE]  # バッチごとに分割

#             # 非同期でバッチ内のファイルをアップロード
#             tasks = [upload_file_async(blob_service_client, file) for file in batch]
#             await asyncio.gather(*tasks)  # 非同期タスクの実行を待機

#             logging.info(f"バッチ {i // BATCH_SIZE + 1} が完了しました。")
#             await asyncio.sleep(2)  # 必要に応じてバッチ間で休止

#         return func.HttpResponse(f"{len(files)} 個のファイルがバッチ処理で非同期アップロードされました。", status_code=200)

#     except Exception as e:
#         logging.error(f"エラーが発生しました: {e}")
#         return func.HttpResponse(f"エラー: {str(e)}", status_code=500)


# # --- 非同期ファイルアップロード処理 ---
# async def upload_file_async(blob_service_client, file):
#     try:
#         # Blob クライアントの非同期取得
#         blob_client = blob_service_client.get_blob_client(container=container_name, blob=file.filename)

#         # ファイルを非同期でアップロード
#         await blob_client.upload_blob(file.stream, timeout=60)
#         logging.info(f"ファイル {file.filename} がアップロードされました。")

#     except ResourceExistsError:
#         logging.warning(f"ファイル {file.filename} はすでに存在します。スキップします。")
#     except Exception as e:
#         logging.error(f"ファイル {file.filename} のアップロード中にエラーが発生しました: {e}")


# --- リクエストに添付されたファイルを処理し、Azure AI Search のインデックスに追加する関数 ---

# @app.route(route="process_file_and_index", auth_level=func.AuthLevel.ANONYMOUS)
# def process_file_and_index(req: func.HttpRequest) -> func.HttpResponse:
#     logging.info('Azure Function for processing files and indexing called.')

#     def get_wikipedia_page(title: str):
#         """
#         Retrieve the full text content of a Wikipedia page.

#         :param title: str - Title of the Wikipedia page.
#         :return: str - Full text content of the page as raw string.
#         """
#         # Wikipedia API endpoint
#         URL = "https://ja.wikipedia.org/w/api.php"

#         # Parameters for the API request
#         params = {
#             "action": "query",
#             "format": "json",
#             "titles": title,
#             "prop": "extracts",
#             "explaintext": True,
#         }

#         # Custom User-Agent header to comply with Wikipedia's best practices
#         headers = {"User-Agent": "tutorial/0.0.1"}

#         response = requests.get(URL, params=params, headers=headers)
#         data = response.json()

#         # Extracting page content
#         page = next(iter(data["query"]["pages"].values()))
#         return page["extract"] if "extract" in page else None

#     full_document = get_wikipedia_page("葬送のフリーレン")


#     try:
       

#         all_texts = []  # すべてのファイルのチャンクを保持するリスト
        
#         for file in files:
#             # 一時ファイルに保存
#             with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#                 temp_file.write(file.stream.read())
#                 temp_file_path = temp_file.name

#             try:
#                 # Document Intelligence でファイルのテキストを抽出
#                 logging.info(f"Processing file: {file.filename}")

#                 # Document Intelligence API の呼び出し
#                 loader = AzureAIDocumentIntelligenceLoader(
#                     file_path=temp_file_path,  # 一時ファイルのパスを渡す
#                     api_key=form_recognizer_key, 
#                     api_endpoint=form_recognizer_endpoint, 
#                     api_model="prebuilt-layout"
#                 )
#                 docs = loader.load()

#                 # 各ドキュメントに対してチャンク処理
#                 for doc in docs:
#                     # ドキュメントからテキストを取得
#                     docs_string = doc.page_content  # リストの各ドキュメントオブジェクトにアクセス

#                     # Markdownベースのセマンティックチャンキング
#                     headers_to_split_on = [
#                         ("#", "Header 1"),
#                         ("##", "Header 2"),
#                         ("###", "Header 3"),
#                     ]

#                     # langchainを活用してセマンティックチャンキングを実施
#                     md_text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
#                     markdown_splits = md_text_splitter.split_text(docs_string)
                    
#                     chunk_size = 500
#                     chunk_overlap = 30
#                     text_splitter = RecursiveCharacterTextSplitter(
#                         chunk_size=chunk_size, chunk_overlap=chunk_overlap
#                     )

#                     # Split the markdown-split text into smaller chunks
#                     splits = text_splitter.split_documents(markdown_splits)

#                     # Documentオブジェクトからpage_contentを取り出し、テキストとして結合
#                     split_texts = [split.page_content for split in splits]
#                     all_texts.append("\n".join(split_texts))

#             finally:
#                 # 一時ファイルを削除
#                 os.remove(temp_file_path)

#         # 全てのファイルのチャンク化されたテキストを結合して返却
#         combined_text = "\n\n".join(all_texts)

#         return func.HttpResponse(
#             body=combined_text,
#             mimetype="text/plain",
#             status_code=200
#         )

#     except Exception as e:
#         logging.error(f"エラーが発生しました: {e}")
#         return func.HttpResponse(f"エラー: {str(e)}", status_code=500)


@app.route(route="process_wikipedia_page", methods=['GET'], auth_level=func.AuthLevel.ANONYMOUS)
def process_wikipedia_page(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Azure Function for processing Wikipedia page and indexing called.')

    def get_wikipedia_page(title: str):
        """
        Retrieve the full text content of a Wikipedia page.

        :param title: str - Title of the Wikipedia page.
        :return: str - Full text content of the page as raw string.
        """
        # Wikipedia API endpoint
        URL = "https://ja.wikipedia.org/w/api.php"

        # Parameters for the API request
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
        }

        # Custom User-Agent header to comply with Wikipedia's best practices
        headers = {"User-Agent": "tutorial/0.0.1"}

        response = requests.get(URL, params=params, headers=headers)
        data = response.json()

        # Extracting page content
        page = next(iter(data["query"]["pages"].values()))
        return page["extract"] if "extract" in page else None

    try:
        # GETリクエストからtitleを取得
        title = req.params.get('title', '葬送のフリーレン')  # 'title' パラメータを取得、デフォルトは"葬送のフリーレン"
        logging.info(f"Fetching Wikipedia page for title: {title}")
        
        # Wikipediaから指定されたページの内容を取得
        full_document = get_wikipedia_page(title)

        if not full_document:
            return func.HttpResponse(f"Wikipedia page '{title}' not found", status_code=404)

        # Markdownベースのセマンティックチャンキング
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        logging.info('セマンティックチャンキング開始')
        # langchainを活用してセマンティックチャンキングを実施
        md_text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        markdown_splits = md_text_splitter.split_text(full_document)

        # チャンク処理 (RecursiveCharacterTextSplitterを使用)
        chunk_size = 500
        chunk_overlap = 30
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # Split the markdown-split text into smaller chunks
        splits = text_splitter.split_documents(markdown_splits)
        logging.info(f'セマンティックチャンキング終了{type(splits)}')

        # Documentオブジェクトからpage_contentを取り出し、区切りを入れて結合
        split_texts = []
        for i, split in enumerate(splits):
            split_texts.append(f"--- チャンク {i+1} ---\n{split.page_content}")

        # チャンク化されたテキストを結合
        combined_text = "\n\n".join(split_texts)

        # チャンク化されたテキストを返却
        return func.HttpResponse(
            body=combined_text,
            mimetype="text/plain",
            status_code=200
        )

    except Exception as e:
        logging.error(f"エラーが発生しました: {e}")
        return func.HttpResponse(f"エラー: {str(e)}", status_code=500)

