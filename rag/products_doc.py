#
# products_doc.py - generates chroma vector database of OpenShift product docs embeddings
#
import time, re, os, pathlib, time
import requests
from tqdm import tqdm
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    DirectoryLoader,
    )

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import chromadb
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
    )

from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

class ProductsDocs:
    def __init__(
        self,
        ocp_version="4.14",
        ocp_sandbox_version="1.5",
        base_url=None,
        base_dir=None,
        ):

        self.url_list = {
            "pdf": [
                {
                    "title": "Assisted Installer for Openshift",
                    "url": "https://access.redhat.com/documentation/en-us/assisted_installer_for_openshift_container_platform/2023/pdf/assisted_installer_for_openshift_container_platform/index",
                },
                {
                    "title": "Sandboxed Containers Release Notes",
                    "url": f"https://access.redhat.com/documentation/en-us/openshift_sandboxed_containers/{ocp_sandbox_version}/pdf/openshift_sandboxed_containers_release_notes/index",
                },
                {
                    "title": "Sandboxed Containers User Guide",
                    "url": f"https://access.redhat.com/documentation/en-us/openshift_sandboxed_containers/{ocp_sandbox_version}/pdf/openshift_sandboxed_containers_user_guide/index",
                },
            ],
        }

        self.ocp_sections = [
            "about",
            "getting_started",
            "release_notes",
            "security_and_compliance",
            "architecture",
            "installing",
            "updating_clusters",
            "authentication_and_authorization",
            "networking",
            "registry",
            "postinstallation_configuration",
            "storage",
            "migration_toolkit_for_containers",
            "backup_and_restore",
            "machine_management",
            "web_console",
            "hosted_control_planes",
            "cli_tools",
            "building_applications",
            "serverless",
            "images",
            "nodes",
            "operators",
            "specialized_hardware_and_driver_enablement",
            "builds_using_buildconfig",
            "logging",
            "monitoring",
            "scalability_and_performance",
            "support",
            "virtualization",
            "distributed_tracing",
            "service_mesh",
        ]

        if base_url is None:
            self.base_url = f"https://access.redhat.com/documentation/en-us/openshift_container_platform/{ocp_version}/pdf/"
        else:
            self.base_url = base_url

        self.base_dir = "docset" if base_dir is None else base_dir
        self.ocp_version=ocp_version
        self.ocp_sandbox_version=ocp_sandbox_version

        self.gen_docs_list()

    def gen_docs_list(self):
        for section in self.ocp_sections:
            section_title = section.replace("_", " ").title()

            self.url_list["pdf"].append({
                "title": section_title, 
                "url": self.base_url + f"{section}/index"
                })


    def pull_docs(self, use_cached=True):
        docs_url = self.url_list["pdf"]
        print(f"Retrieving {len(docs_url)} documents")

        for entry in tqdm(docs_url, 
                          desc="fetching document",
                          ):
            title = entry["title"]
            url = entry["url"]
            base_dir_path=self.base_dir+f"/{self.ocp_version}/"
            fname = base_dir_path+title.replace(" ","_")+".pdf"

            if not os.path.exists(base_dir_path):
                os.makedirs(base_dir_path)

            if os.path.isfile(fname) and use_cached:
                #print(f"Skipping downloading {url}. File {fname} already exist.")
                pass
            else:
                response = requests.get(url, stream=True)
                with open(fname, mode="wb") as file:
                    for chunk in response.iter_content():
                        file.write(chunk)

            # TODO: confirm the resulting document is a valid PDF

    def get_cached_docs(self, ocp_sections=None):
        base_dir_path = f"{self.base_dir}/{self.ocp_version}/"
        docs_path = pathlib.Path(base_dir_path)
        if ocp_sections is not None and isinstance(ocp_sections, list):
            file_list = list(map(lambda x: f"{docs_path}/{str(x).title()}.pdf", ocp_sections))
        else:
            file_list = list(map(str,docs_path.rglob("*.pdf")))
        return file_list


def embedding_function(model_name=None):
    # hardcoded provider & model for embeddings
    match model_name:
        case 'mistral':
            embeddings = OllamaEmbeddings(
                base_url="http://localhost:11434", model="mistral:latest"
            )
        case 'phi2':
            embeddings = OllamaEmbeddings(
                base_url="http://localhost:11434", model="phi:latest"
            )
        case 'openai':
            embeddings = OpenAIEmbeddings()
        case _:
            # __Model_Name__        __Dimensions__   __Usages__
            # all-MiniLM-L6-v2        384            clustering & semantic search
            # e5-large-v2             1024           clustering & semantic search
            # BAAI/bge-large-en-v1.5  1024           retrieval, classification, clustering & semantic search
            # all-roberta-large-v1    1024           clustering & semantic search
            # https://huggingface.co/spaces/mteb/leaderboard 
            # https://www.sbert.net/docs/pretrained_models.html 
            # https://huggingface.co/sentence-transformers 
            default_model = "BAAI/bge-large-en-v1.5"
            if model_name is not None:
                print(f"Warning: Unsupported model {str(model_name)}. Using default embedding model {default_model}")
            # create the open-source embedding function
            embeddings = SentenceTransformerEmbeddings(model_name=default_model)
    return embeddings


def replace_ligatures(text: str) -> str:
    """
    https://pypdf.readthedocs.io/en/latest/user/post-processing-in-text-extraction.html#ligature-replacement
    """
    ligatures = {
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬅ": "ft",
        "ﬆ": "st",
        # "Ꜳ": "AA",
        # "Æ": "AE",
        "ꜳ": "aa",
    }
    for search, replace in ligatures.items():
        text = text.replace(search, replace)
    return text

def create_embeddings(collection_name,
                      ocp_version="4.14",
                      ocp_sandbox_version="1.5",
                      chunk_size=1024,
                      chunk_overlap=0,
                      embedding_provider=None,
                      ocp_sections=None,
                      ):
    # To calculate full run 
    start_run = time.time()

    # remove collection if exists
    try:
        vectordb = chromadb.PersistentClient(path="chroma")
        vectordb.delete_collection(collection_name)
    except:
        pass

    # provider & model for embeddings
    embeddings = embedding_function(embedding_provider)
    if embeddings is None:
        print(f"ERROR: Undefined embedding function")
        return
    
    # create new ccollection with custom embeddings
    vectordb = Chroma(
        # collection name is the same as name of model_name
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory="chroma",
        )

    # use large chunk size for "understanding" text for summarization and Q&A
    # using large chunk overlaps for better sliding windows on retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap
                                                   )

    prod_docs = ProductsDocs(
        ocp_version=ocp_version, ocp_sandbox_version=ocp_sandbox_version
    )
    prod_docs.pull_docs()       # download prod docs to local destination

    for fname in tqdm(prod_docs.get_cached_docs(ocp_sections),
                      desc="Loading document"):
        
        print(f"{'#'*40}\nLoading {fname}...")
        retrieval_timestamp = time.asctime()  # get timestamp once for the doc
        try:
            document = PyPDFLoader(fname).load()
        except Exception as e:
            print("WARNING. Something went wrong processig {fname}. Continuing with the next document.\n Exception = {e}")
            continue

        pages_total = document[-1].metadata['page']
        print(f"Found {pages_total} pages in document")

        for page in tqdm(document,
                        desc="Cleaning pages",
                        ):
            title = page.metadata['source'].split("/")[-1].split(".")[0].replace("_", " ").title()
            page.metadata["title"] = title
            url=f"https://access.redhat.com/documentation/en-us/openshift_container_platform/{ocp_version}/"
            page.metadata["url"]=url
            page.metadata["ocp_version"] = ocp_version
            page.metadata["retrieval_timestamp"] = retrieval_timestamp
                        
            try:
                ##print(f"Processing '{title}' page {page.metadata['page']}...")
                # TODO: consider the use of ParentDocumentRetriever
                # TODO: generate summarization as child
                content = page.page_content
                # remove multiple empty spaces with a single space
                # WARNING: removing extra spaces breaks YAML and other preformated context
                content = re.sub("[ ]{2,}", " ", content)   # see note above
                content = re.sub(".[ .]{2,}", ".", content)
                content = re.sub("[.\n]{2,}", "\n", content)
                content = re.sub("[\n]{2,}", "\n", content)
                content = replace_ligatures(content)
                #
                page.page_content = content

            except:
                print(f"ERROR processing [{title}] page {page.metadata['page']}. Continuing with next document.")
                continue

        docs_chunks = text_splitter.split_documents(document)
        print(f"Embedding {len(docs_chunks)} chunks into vector database...")
        start = time.time()
        vectordb.add_documents(docs_chunks)
        end = time.time()
        vectordb.persist()
        print(f"Embedding process took {end - start:.2f} seconds...\n")

    # To calculate full run
    end_run = time.time()
    print(f"Embedding all documents took {(end_run - start_run)/60:.2f} minutes...\n")


if __name__ == "__main__":
    create_embeddings(collection_name='OpenShift', 
                      embedding_provider=None,
                      ocp_sections=None)


# Embedding times comparisson 
# (local=BAAI/bge-large-en-v1.5)
# Pages local   phi2    mistral
# 109   53.80   92.28   185.78
# 148   114.19  147.89  289.28
# 162   97.50   154.55  313.71
