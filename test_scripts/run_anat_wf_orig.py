from fileformats.medimage import NiftiGzX, T1Weighted
import logging
from pathlib import Path
from logging import DEBUG, FileHandler

# from niworkflows.utils.bids import DEFAULT_BIDS_QUERIES, collect_data
from mriqc._warnings import DATE_FMT, LOGGER_FMT, _LogFormatter
import atexit
import time
import tempfile
from mriqc import config
from mriqc.cli.parser import parse_args

from mriqc.workflows.anatomical.base import anat_qc_workflow

# from mriqc import config


# class Execution:
#     log_dir = "/Users/tclose/Data/pydra-mriqc-test2.log"


class opts:
    output_dir = "/Users/tclose/Data/pydra-mriqc-test2"
    verbose = 0
    species = "human"
    modalities = config.SUPPORTED_SUFFIXES
    bids_database_wipe = False
    testing = False
    float32 = True
    pdb = False
    work_dir = Path("work").absolute()
    verbose_reports = False
    reports_only = False
    write_graph = False
    dry_run = False
    profile = False
    use_plugin = None
    no_sub = False
    email = ""
    upload_strict = False
    ants_float = False
    # Diffusion workflow settings
    min_dwi_length = config.workflow.min_len_dwi
    min_bold_length = config.workflow.min_len_bold
    fft_spikes_detector = False
    fd_thres = 0.2
    deoblique = False
    despike = False
    verbose_count = 0


config.execution.log_level = int(max(25 - 5 * opts.verbose_count, DEBUG))

config.loggers.init()

_log_file = Path(opts.output_dir) / "logs" / f"mriqc-{config.execution.run_uuid}.log"
_log_file.parent.mkdir(exist_ok=True, parents=True)
_handler = FileHandler(_log_file)
_handler.setFormatter(
    _LogFormatter(
        fmt=LOGGER_FMT.format(color="", reset=""),
        datefmt=DATE_FMT,
        colored=False,
    )
)
config.loggers.default.addHandler(_handler)

extra_messages = [""]


# config.loggers.cli.log(
#     26,
#     PARTICIPANT_START.format(
#         version=__version__,
#         bids_dir=opts.bids_dir,
#         output_dir=opts.output_dir,
#         analysis_level=opts.analysis_level,
#         extra_messages="\n".join(extra_messages),
#     ),
# )
# config.from_dict(vars(opts))

# Load base plugin_settings from file if --use-plugin
if opts.use_plugin is not None:
    from yaml import safe_load as loadyml

    with open(opts.use_plugin) as f:
        plugin_settings = loadyml(f)
    _plugin = plugin_settings.get("plugin")
    if _plugin:
        config.nipype.plugin = _plugin
        config.nipype.plugin_args = plugin_settings.get("plugin_args", {})
        config.nipype.nprocs = config.nipype.plugin_args.get(
            "nprocs", config.nipype.nprocs
        )

# # Load BIDS filters
# if opts.bids_filter_file:
#     config.execution.bids_filters = loads(opts.bids_filter_file.read_text())

# bids_dir = config.execution.bids_dir
config.execution.output_dir = Path("/Users/tclose/Data/pydra-mriqc-test2-output")
output_dir = config.execution.output_dir
work_dir = config.execution.work_dir
version = config.environment.version

# config.execution.bids_dir_datalad = (
#     config.execution.datalad_get
#     and (bids_dir / ".git").exists()
#     and (bids_dir / ".datalad").exists()
# )

# Setup directories
config.execution.log_dir = output_dir / "logs"
# Check and create output and working directories
config.execution.log_dir.mkdir(exist_ok=True, parents=True)
output_dir.mkdir(exist_ok=True, parents=True)
work_dir.mkdir(exist_ok=True, parents=True)

# Force initialization of the BIDSLayout
# config.execution.init()

# participant_label = [
#     d.name[4:]
#     for d in config.execution.bids_dir.glob("sub-*")
#     if d.is_dir() and d.exists()
# ]


config.execution.participant_label = "sub-01"

# Handle analysis_level
analysis_level = set(config.workflow.analysis_level)
if not config.execution.participant_label:
    analysis_level.add("group")
config.workflow.analysis_level = list(analysis_level)

# List of files to be run
lc_modalities = "t1w"  # [mod.lower() for mod in config.execution.modalities]
# bids_dataset, _ = collect_data(
#     config.execution.layout,
#     config.execution.participant_label,
#     session_id=config.execution.session_id,
#     task=config.execution.task_id,
#     group_echos=True,
#     bids_filters={
#         mod: config.execution.bids_filters.get(mod, {}) for mod in lc_modalities
#     },
#     queries={mod: DEFAULT_BIDS_QUERIES[mod] for mod in lc_modalities},
# )

# Drop empty queries
# bids_dataset = {mod: files for mod, files in bids_dataset.items() if files}
config.workflow.inputs = None  # bids_dataset


# set specifics for alternative populations
if opts.species.lower() != "human":
    config.workflow.species = opts.species
    # TODO: add other species once rats are working
    if opts.species.lower() == "rat":
        config.workflow.template_id = "Fischer344"
        # mean distance from the lateral edge to the center of the brain is
        # ~ PA:10 mm, LR:7.5 mm, and IS:5 mm (see DOI: 10.1089/089771503770802853)
        # roll movement is most likely to occur, so set to 7.5 mm
        config.workflow.fd_radius = 7.5
        # block uploads for the moment; can be reversed before wider release
        config.execution.no_sub = True


log_file = Path("/Users/tclose/Data/pydra-mriqc-test.log")
log_file.unlink(missing_ok=True)

pydra_logger = logging.getLogger("pydra")
pydra_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(str(log_file))
pydra_logger.addHandler(file_handler)
pydra_logger.addHandler(logging.StreamHandler())

tmp_dir = Path(tempfile.mkdtemp())

t1w = NiftiGzX[T1Weighted].sample()
t1w = t1w.copy(tmp_dir, new_stem="sub-01_T1w")


workflow = anat_qc_workflow(in_file=str(t1w), metadata=t1w.metadata)
workflow.run()
# workflow.cache_dir = "/Users/tclose/Data/pydra-mriqc-test-cache2"
result = workflow(plugin="serial")
print(result.out)

atexit.register(config.restore_env)

config.settings.start_time = time.time()

# Run parser
parse_args()

# if config.execution.pdb:
#     from mriqc.utils.debug import setup_exceptionhook

#     setup_exceptionhook()
#     config.nipype.plugin = "Linear"

# # CRITICAL Save the config to a file. This is necessary because the execution graph
# # is built as a separate process to keep the memory footprint low. The most
# # straightforward way to communicate with the child process is via the filesystem.
# # The config file name needs to be unique, otherwise multiple mriqc instances
# # will create write conflicts.
# config_file = config.to_filename()
# config.loggers.cli.info(f"MRIQC config file: {config_file}.")

# exitcode = 0
# # Set up participant level
# _pool = None
# if config.nipype.plugin in ("MultiProc", "LegacyMultiProc"):
#     import multiprocessing as mp
#     import multiprocessing.forkserver
#     from concurrent.futures import ProcessPoolExecutor
#     from contextlib import suppress

#     os.environ["OMP_NUM_THREADS"] = "1"
#     os.environ["NUMEXPR_MAX_THREADS"] = "1"

#     with suppress(RuntimeError):
#         mp.set_start_method("fork")
#     gc.collect()

#     _pool = ProcessPoolExecutor(
#         max_workers=config.nipype.nprocs,
#         initializer=config._process_initializer,
#         initargs=(config_file,),
#     )

# _resmon = None
# if config.execution.resource_monitor:
#     from mriqc.instrumentation.resources import ResourceRecorder

#     _resmon = ResourceRecorder(
#         pid=os.getpid(),
#         log_file=mkstemp(
#             dir=config.execution.work_dir, prefix=".resources.", suffix=".tsv"
#         )[1],
#     )
#     _resmon.start()

# if not config.execution.notrack:
#     from ..utils.telemetry import setup_migas

#     setup_migas()

# with Manager() as mgr:
#     from .workflow import build_workflow

#     retval = mgr.dict()
#     p = Process(target=build_workflow, args=(str(config_file), retval))
#     p.start()
#     p.join()

#     mriqc_wf = retval.get("workflow", None)
#     exitcode = p.exitcode or retval.get("return_code", 0)

# CRITICAL Load the config from the file. This is necessary because the ``build_workflow``
# function executed constrained in a process may change the config (and thus the global
# state of MRIQC).
# config.load(config_file)

# exitcode = exitcode or (mriqc_wf is None) * os.EX_SOFTWARE
# if exitcode != 0:
#     sys.exit(exitcode)

# # Initialize nipype config
# config.nipype.init()
# # Make sure loggers are started
# config.loggers.init()

# if _resmon:
#     config.loggers.cli.info(f"Started resource recording at {_resmon._logfile}.")

# # Resource management options
# if config.nipype.plugin in ("MultiProc", "LegacyMultiProc") and (
#     1 < config.nipype.nprocs < config.nipype.omp_nthreads
# ):
#     config.loggers.cli.warning(
#         "Per-process threads (--omp-nthreads=%d) exceed total "
#         "threads (--nthreads/--n_cpus=%d)",
#         config.nipype.omp_nthreads,
#         config.nipype.nprocs,
#     )

# # Check synthstrip is properly installed
# if not config.environment.synthstrip_path:
#     config.loggers.cli.warning(
#         (
#             "Please make sure FreeSurfer is installed and the FREESURFER_HOME "
#             "environment variable is defined and pointing at the right directory."
#         )
#         if config.environment.freesurfer_home is None
#         else (
#             f"FreeSurfer seems to be installed at {config.environment.freesurfer_home},"
#             " however SynthStrip's model is not found at the expected path."
#         )
#     )

# if mriqc_wf is None:
#     sys.exit(os.EX_SOFTWARE)

# if mriqc_wf and config.execution.write_graph:
#     mriqc_wf.write_graph(graph2use="colored", format="svg", simple_form=True)

# if not config.execution.dry_run and not config.execution.reports_only:
#     # Warn about submitting measures BEFORE
#     if not config.execution.no_sub:
#         config.loggers.cli.warning(config.DSA_MESSAGE)

#     # Clean up master process before running workflow, which may create forks
#     gc.collect()
#     # run MRIQC
#     _plugin = config.nipype.get_plugin()
#     if _pool:
#         MultiProcPlugin

#         _plugin = {
#             "plugin": MultiProcPlugin(
#                 pool=_pool, plugin_args=config.nipype.plugin_args
#             ),
#         }
#     mriqc_wf.run(**_plugin)

#     # Warn about submitting measures AFTER
#     if not config.execution.no_sub:
#         config.loggers.cli.warning(config.DSA_MESSAGE)

# if not config.execution.dry_run:
#     from mriqc.reports.individual import generate_reports

#     generate_reports()

# _subject_duration = time.gmtime(
#     (time.time() - config.settings.start_time)
#     / sum(len(files) for files in config.workflow.inputs.values())
# )
# config.loggers.cli.log(
#     25,
#     messages.PARTICIPANT_FINISHED.format(
#         duration=time.strftime("%Hh %Mmin %Ss", _subject_duration)
#     ),
# )

# if _resmon is not None:
#     from mriqc.instrumentation.viz import plot

#     _resmon.stop()
#     plot(
#         _resmon._logfile,
#         param="mem_rss_mb",
#         out_file=str(_resmon._logfile).replace(".tsv", ".rss.png"),
#     )
#     plot(
#         _resmon._logfile,
#         param="mem_vsm_mb",
#         out_file=str(_resmon._logfile).replace(".tsv", ".vsm.png"),
#     )
