from __future__ import print_function, division, absolute_import
from contextlib import contextmanager
import warnings

from numba import ir
import weakref
from collections import namedtuple, deque
import operator

from numba.lowering import Lower, _VarArgItem

from llvmlite.llvmpy.core import Constant, Type, Builder
from numba.stencilparfor import StencilPass

from numba import (
    config,
    errors,
    types,
    rewrites,
    typeinfer,
    funcdesc,
    utils,
    typing
    )

from numba.errors import (LoweringError, new_error_context, TypingError,
                     LiteralTypingError)

from numba.compiler_machinery import FunctionPass, LoweringPass, register_pass

from .dppy_lowerer import DPPyLower

from numba.parfor import PreParforPass as _parfor_PreParforPass
from numba.parfor import ParforPass as _parfor_ParforPass
from numba.parfor import Parfor


@register_pass(mutates_CFG=True, analysis_only=False)
class DPPyConstantSizeStaticLocalMemoryPass(FunctionPass):

    _name = "dppy_constant_size_static_local_memory_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Preprocessing for data-parallel computations.
        """
        # Ensure we have an IR and type information.
        assert state.func_ir
        func_ir = state.func_ir

        _DEBUG = False

        if _DEBUG:
            print('Checks if size of OpenCL local address space alloca is a compile-time constant.'.center(80, '-'))
            print(func_ir.dump())

        work_list = list(func_ir.blocks.items())
        while work_list:
            label, block = work_list.pop()
            for i, instr in enumerate(block.body):
                if isinstance(instr, ir.Assign):
                    expr = instr.value
                    if isinstance(expr, ir.Expr):
                        if expr.op == 'call':
                            call_node = block.find_variable_assignment(expr.func.name).value
                            if isinstance(call_node, ir.Expr) and call_node.attr == "static_alloc":
                                arg = None
                                # at first look in keyword arguments to get the shape, which has to be
                                # constant
                                if expr.kws:
                                    for _arg in expr.kws:
                                        if _arg[0] == "shape":
                                            arg = _arg[1]

                                if not arg:
                                    arg = expr.args[0]

                                error = False
                                # arg can be one constant or a tuple of constant items
                                arg_type = func_ir.get_definition(arg.name)
                                if isinstance(arg_type, ir.Expr):
                                    # we have a tuple
                                    for item in arg_type.items:
                                        if not isinstance(func_ir.get_definition(item.name), ir.Const):
                                            error = True
                                            break

                                else:
                                    if not isinstance(func_ir.get_definition(arg.name), ir.Const):
                                        error = True
                                        break

                                if error:
                                    warnings.warn_explicit("The size of the Local memory has to be constant",
                                                           errors.NumbaError,
                                                           state.func_id.filename,
                                                           state.func_id.firstlineno)
                                    raise



        if config.DEBUG or config.DUMP_IR:
            name = state.func_ir.func_id.func_qualname
            print(("IR DUMP: %s" % name).center(80, "-"))
            state.func_ir.dump()

        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class DPPyPreParforPass(FunctionPass):

    _name = "dppy_pre_parfor_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Preprocessing for data-parallel computations.
        """
        # Ensure we have an IR and type information.
        assert state.func_ir

        preparfor_pass = _parfor_PreParforPass(
            state.func_ir,
            state.type_annotation.typemap,
            state.type_annotation.calltypes, state.typingctx,
            state.flags.auto_parallel,
            state.parfor_diagnostics.replaced_fns
        )

        preparfor_pass.run()

        if config.DEBUG or config.DUMP_IR:
            name = state.func_ir.func_id.func_qualname
            print(("IR DUMP: %s" % name).center(80, "-"))
            state.func_ir.dump()

        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class DPPyParforPass(FunctionPass):

    _name = "dppy_parfor_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Convert data-parallel computations into Parfor nodes
        """
        # Ensure we have an IR and type information.
        assert state.func_ir
        parfor_pass = _parfor_ParforPass(state.func_ir,
                                         state.type_annotation.typemap,
                                         state.type_annotation.calltypes,
                                         state.return_type,
                                         state.typingctx,
                                         state.flags.auto_parallel,
                                         state.flags,
                                         state.parfor_diagnostics)

        parfor_pass.run()

        if config.DEBUG or config.DUMP_IR:
            name = state.func_ir.func_id.func_qualname
            print(("IR DUMP: %s" % name).center(80, "-"))
            state.func_ir.dump()

        return True


@contextmanager
def fallback_context(state, msg):
    """
    Wraps code that would signal a fallback to object mode
    """
    try:
        yield
    except Exception as e:
        if not state.status.can_fallback:
            raise
        else:
            if utils.PYVERSION >= (3,):
                # Clear all references attached to the traceback
                e = e.with_traceback(None)
            # this emits a warning containing the error message body in the
            # case of fallback from npm to objmode
            loop_lift = '' if state.flags.enable_looplift else 'OUT'
            msg_rewrite = ("\nCompilation is falling back to object mode "
                           "WITH%s looplifting enabled because %s"
                           % (loop_lift, msg))
            warnings.warn_explicit('%s due to: %s' % (msg_rewrite, e),
                                   errors.NumbaWarning,
                                   state.func_id.filename,
                                   state.func_id.firstlineno)
            raise


@register_pass(mutates_CFG=True, analysis_only=False)
class SpirvFriendlyLowering(LoweringPass):

    _name = "spirv_friendly_lowering"

    def __init__(self):
        LoweringPass.__init__(self)

    def run_pass(self, state):
        if state.library is None:
            codegen = state.targetctx.codegen()
            state.library = codegen.create_library(state.func_id.func_qualname)
            # Enable object caching upfront, so that the library can
            # be later serialized.
            state.library.enable_object_caching()

        targetctx = state.targetctx
        library   = state.library
        interp    = state.func_ir  # why is it called this?!
        typemap   = state.typemap
        restype   = state.return_type
        calltypes = state.calltypes
        flags     = state.flags
        metadata  = state.metadata

        msg = ("Function %s failed at nopython "
               "mode lowering" % (state.func_id.func_name,))
        with fallback_context(state, msg):
            # Lowering
            fndesc = \
                funcdesc.PythonFunctionDescriptor.from_specialized_function(
                    interp, typemap, restype, calltypes,
                    mangler=targetctx.mangler, inline=flags.forceinline,
                    noalias=flags.noalias)

            with targetctx.push_code_library(library):
                lower = DPPyLower(targetctx, library, fndesc, interp,
                                       metadata=metadata)
                lower.lower()
                if not flags.no_cpython_wrapper:
                    lower.create_cpython_wrapper(flags.release_gil)
                env = lower.env
                call_helper = lower.call_helper
                del lower

            from numba.compiler import _LowerResult  # TODO: move this
            if flags.no_compile:
                state['cr'] = _LowerResult(fndesc, call_helper,
                                           cfunc=None, env=env)
            else:
                # Prepare for execution
                cfunc = targetctx.get_executable(library, fndesc, env)
                # Insert native function for use by other jitted-functions.
                # We also register its library to allow for inlining.
                targetctx.insert_user_function(cfunc, fndesc, [library])
                state['cr'] = _LowerResult(fndesc, call_helper,
                                           cfunc=cfunc, env=env)
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class DPPyNoPythonBackend(FunctionPass):

    _name = "nopython_backend"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Back-end: Generate LLVM IR from Numba IR, compile to machine code
        """
        lowered = state['cr']
        signature = typing.signature(state.return_type, *state.args)

        from numba.compiler import compile_result
        state.cr = compile_result(
            typing_context=state.typingctx,
            target_context=state.targetctx,
            entry_point=lowered.cfunc,
            typing_error=state.status.fail_reason,
            type_annotation=state.type_annotation,
            library=state.library,
            call_helper=lowered.call_helper,
            signature=signature,
            objectmode=False,
            interpmode=False,
            lifted=state.lifted,
            fndesc=lowered.fndesc,
            environment=lowered.env,
            metadata=state.metadata,
            reload_init=state.reload_init,
        )
        return True
